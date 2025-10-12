# query_chroma_systemprompt_b64.py
import os, tempfile, time, base64, requests, numpy as np, sounddevice as sd, soundfile as sf
from typing import List, Optional
from faster_whisper import WhisperModel
from piper import PiperVoice
import chromadb
from sentence_transformers import SentenceTransformer
import cv2, time
from collections import Counter
from ultralytics import YOLO
from ultralytics import YOLO
import os, tempfile
# ---------- CONFIG ----------
LMSTUDIO_URL   = "http://127.0.0.1:1234/v1/chat/completions"
LMSTUDIO_MODEL = "astral_cortex"
TTS_VOICE      = r"C:\memcore\voice.onnx"
YOLO_MODEL_PATH = "yolov8x.pt"
#yolo_model = YOLO(YOLO_MODEL_PATH)
cam = cv2.VideoCapture(0)
ASR_MODEL      = "tiny.en"
SAMPLE_RATE    = 16000
CHANNELS       = 1
RECORD_SECONDS = 3
CHROMA_PATH        = r"C:\memcore\chromadb_data"
CHROMA_COLLECTION  = None
TOP_K              = 3
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEM_PROMPT_PATH = r"C:\memcore\system_prompt.jar"
# ---------------------------

# ---------- VISION CONFIG ----------
YOLO_MODELS = {
    "object": "8n.pt",
    "holding a": "v8_detect_guns.pt",
}

print("🔧 Loading YOLO models...")
models = {}
for name, path in YOLO_MODELS.items():
    try:
        models[name] = YOLO(path)
        print(f"✅ Loaded {name}: {path}")
    except Exception as e:
        print(f"⚠️ Could not load {name}: {e}")
print(f"Total models loaded: {len(models)}")
cam = cv2.VideoCapture(0)
# -----------------------------------
def get_multi_vision_summary(conf=0.4):
    """Run multiple YOLO models on a single frame and combine results."""
    try:
        if not cam.isOpened():
            raise RuntimeError("Camera not accessible.")
        ok, frame = cam.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture frame.")

        all_detections = {}
        for name, model in models.items():
            try:
                res = model.predict(frame, conf=conf, verbose=False)[0]
                names = model.model.names if hasattr(model.model, "names") else model.names
                labels = [names[int(b.cls)] for b in res.boxes]
                if labels:
                    all_detections[name] = Counter(labels)
            except Exception as inner:
                print(f"⚠️ {name} detection error: {inner}")

        if not all_detections:
            return "You are currently seeing: nothing."

        summary_parts = []
        for name, objs in all_detections.items():
            joined = ", ".join([f"{k}×{v}" for k, v in objs.items()])
            summary_parts.append(f"{name}: {joined}")
        summary_text = " | ".join(summary_parts)
        return f"You are currently seeing: {summary_text}."

    except Exception as e:
        print("⚠️ Vision error:", e)
        return "You are currently seeing: nothing."
    
print("Loading ASR + TTS models...")
whisper = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
voice   = PiperVoice.load(TTS_VOICE)

print(f"Connecting to Chroma (persistent) at: {CHROMA_PATH}")
client  = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(name: Optional[str] = CHROMA_COLLECTION):
    if name:
        return client.get_collection(name=name)
    cols = client.list_collections()
    if not cols:
        raise RuntimeError(f"No collections found in {CHROMA_PATH}")
    return client.get_collection(name=cols[0].name)

col = get_collection()
print(f"Using collection: {col.name}")

embedder = SentenceTransformer(EMBED_MODEL)

def load_system_prompt_b64(path=SYSTEM_PROMPT_PATH):
    if not os.path.exists(path):
        print(f"⚠️ No system prompt found at {path}, using fallback.")
        return "You are AstraMech. Be concise and factual."
    try:
        with open(path, "r", encoding="utf-8") as f:
            encoded = f.read().strip()
        decoded_bytes = base64.b64decode(encoded)
        decoded_text = decoded_bytes.decode("utf-8", errors="ignore").strip()
        return decoded_text
    except Exception as e:
        print(f"⚠️ Error decoding system prompt: {e}")
        return "You are AstraMech. Be concise and factual."


def record_wav(path, seconds=RECORD_SECONDS):
    print(f"🎙 Recording {seconds}s...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="float32")
    sd.wait()
    sf.write(path, audio, SAMPLE_RATE)
    print("✅ Recorded")


def transcribe(path):
    print("📝 Transcribing...")
    segments, _ = whisper.transcribe(path, beam_size=1, vad_filter=True)
    text = "".join(s.text for s in segments).strip()
    print(f"👂 Heard: {text or '[EMPTY]'}")
    return text


def retrieve_context(query: str, k: int = TOP_K):
    print("📚 Retrieving context from Chroma...")
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=qvec, n_results=k,
                    include=["documents", "metadatas", "distances"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return list(zip(docs, metas, dists))


def compose_prompt(user_query: str, hits: List[tuple]):
    ctx_lines = []
    for i, (doc, meta, dist) in enumerate(hits, 1):
        tag = (meta or {}).get("source") or (meta or {}).get("file") or (meta or {}).get("id") or f"chunk_{i}"
        ctx_lines.append(f"[{tag}] {doc}")
    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context retrieved)"

    sys_prompt = load_system_prompt_b64()

    full_prompt = (
        f"{sys_prompt}\n\n"
        f"# Context\n{context_block}\n\n"
        f"# User said:{user_query}\n\n"
        f"# If Context has no information about this, say 'I do not know.'\n\n"
        f"# Answer the user directly and concisely below using straight sentences:\n"
    )
    return full_prompt

def ask_lmstudio(user_query: str):
    hits   = retrieve_context(user_query, k=TOP_K)
    prompt = compose_prompt(user_query, hits)
    print("💬 Asking LM Studio...")
    r = requests.post(
        LMSTUDIO_URL,
        headers={"Authorization": "Bearer lm-studio"},
        json={"model": LMSTUDIO_MODEL, "messages": [{"role": "user", "content": prompt}]},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def speak(text, out_path=None, play=True):
    print("🗣 Speaking (streaming)...")
    bufs, sr, ch = [], None, 1
    for chunk in voice.synthesize(text):
        arr = getattr(chunk, "audio_float_array", None)
        sr  = getattr(chunk, "sample_rate", None)
        ch  = getattr(chunk, "sample_channels", 1)
        if arr is None or sr is None:
            arr = getattr(chunk, "audio", arr)
            sr  = getattr(chunk, "sampling_rate", sr)
        if arr is None or sr is None:
            raise AttributeError(f"Unknown Piper chunk format: {dir(chunk)}")
        if ch and ch > 1 and arr.ndim == 1:
            arr = arr.reshape(-1, ch)
        bufs.append(arr)
        if play:
            sd.play(arr, sr); sd.wait()
    if not bufs: return
    full = np.concatenate(bufs, axis=0)
    if out_path:
        sf.write(out_path, full, sr, subtype="PCM_16")
    print("✅ Speech done.")

# -------- loop --------
try:
    print("Ready! Press Enter to talk, Ctrl+C to exit.")
    while True:
        input("\n🔴 Press Enter to record…")

        # Windows-safe temp wav path
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # release handle so soundfile can open it

        try:
            record_wav(wav_path)           # writes the mic audio via sf.write(...)
            q = transcribe(wav_path)
        finally:
            # ensure no residual temp file
            try: os.remove(wav_path)
            except FileNotFoundError: pass

        if not q:
            print("…no speech detected, try again.")
            continue

        vision_desc = get_multi_vision_summary()
        print(f"👁 Vision: {vision_desc}")
        q_with_vision = f"{q}\n\n{vision_desc}"
        reply = ask_lmstudio(q_with_vision)
        print(f"🤖 LM Studio: {reply}")

        # Stream Piper audio only; don't save a reply_*.wav
        speak(reply, out_path=None, play=True)

except KeyboardInterrupt:
    print("\nBye! 👋")