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
import threading, sys, time
# ---------- CONFIG ----------
LMSTUDIO_URL   = "http://127.0.0.1:1234/v1/chat/completions"
LMSTUDIO_MODEL = "astral_cortex"
TTS_VOICE      = r"C:\memcore\voice.onnx"
ASR_MODEL      = "tiny.en"
SAMPLE_RATE    = 16000
CHANNELS       = 1
RECORD_SECONDS = 3
CHROMA_PATH        = r"C:\memcore\chromadb_data"
CHROMA_COLLECTION  = None
TOP_K              = 3
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEM_PROMPT_PATH = r"C:\memcore\system_prompt.txt"
POLL_INTERVAL      = 5     
AUDIO_RMS_THRESH   = 0.008  
MIN_SILENCE_BYPASS = False
MIN_VISUAL_OBJECTS = 1  # require at least this many detections to count as "visual activity"
OBSERVER_MODE = False   # only talk when there's something (audio OR visual)
CHATTER_MODE  = True  # talk even on manual trigger; still skips if you want (see logic below)

YOLO_MODELS = {
    "object": "8n.pt",
    "holding a": "v8_detect_guns.pt",
}
# --------------------------------
if OBSERVER_MODE and CHATTER_MODE:
    raise ValueError("Choose only one: set either OBSERVER_MODE or CHATTER_MODE to True, not both.")
enter_event = threading.Event()

def _enter_listener():
    while True:
        try:
            _ = sys.stdin.readline()  
        except Exception:
            break 
        enter_event.set()

threading.Thread(target=_enter_listener, daemon=True).start()

def wait_or_enter(timeout_sec: float) -> bool:
    """Wait up to timeout_sec; return True if Enter was pressed."""
    print(f"Waiting {timeout_sec}s or press Enter to captureâ€¦", end="\r", flush=True)
    triggered = enter_event.wait(timeout_sec)
    if triggered:
        enter_event.clear()
    return triggered

def compute_rms(x: np.ndarray) -> float:
    if x is None or x.size == 0:
        return 0.0
    x = x.astype("float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return float(np.sqrt(np.mean(np.square(x))))

# Listen
def capture_audio_to_temp(seconds=RECORD_SECONDS):
    """Record audio, write to a temp wav, return (wav_path, rms, duration_sec)."""
    print(f"⏺️ Auto-recording {seconds} ⚙️")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="float32")
    sd.wait()
    rms = compute_rms(audio)
    fd, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(wav_path, audio, SAMPLE_RATE)
    print(f"✍🏻 Recorded (RMS={rms:.5f})")
    return wav_path, rms, seconds

# Open eyes
def open_camera(index=0, backend=None):
    """
    Opens the webcam safely on Windows and Linux.
    Tries multiple backends so it's less likely to fail.
    """
    if backend is None and os.name == "nt":
        backend = cv2.CAP_DSHOW

    cap = cv2.VideoCapture(index if backend is None else index, backend or 0)
    if not cap.isOpened():
        if os.name == "nt" and backend != cv2.CAP_MSMF:
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

    if not cap.isOpened():
        raise RuntimeError("âŒ Camera not accessible. Check if another app is using it.")

    print("📷 Warming up camera (2s) 🛠️", end="", flush=True)
    start_time = time.time()
    while time.time() - start_time < 2.0:
        cap.read()
        time.sleep(0.05)
    print(" done!")

    return cap

# Warmup eyes
def warmup_camera(cap, frames=10, delay=0.1):
    """Discard first few frames so exposure and focus settle."""
    print("📷 Warming up camera ⚙️", end="", flush=True)
    for _ in range(frames):
        cap.read()
        time.sleep(delay)
    print(" done!")

cam = open_camera()
warmup_camera(cam, frames=20, delay=0.05)  

def wait_for_enter_blocking():
    print("\n🔴 Press Enter to capture…", flush=True)
    enter_event.clear()
    enter_event.wait()   # blocks until Enter is pressed

# Check eyes
def has_visual_activity(summary_text: str) -> bool:
    """Return True if the YOLO summary indicates something was detected."""
    return "nothing" not in summary_text.lower()

# Check ears
def has_audio_activity(rms: float, transcript: str) -> bool:
    """Audio is present if RMS crosses threshold OR Whisper heard something (configurable)."""
    heard_text = bool((transcript or "").strip())
    loud_enough = rms >= AUDIO_RMS_THRESH
    return (heard_text or loud_enough) if MIN_SILENCE_BYPASS else loud_enough

# Load eye brainstuff
print("👁️ Loading YOLO models...")
models = {}
for name, path in YOLO_MODELS.items():
    try:
        models[name] = YOLO(path)
        print(f"⚙️ Loaded {name}: {path}")
    except Exception as e:
        print(f"❌ Could not load {name}: {e}")
print(f"👜 Total models loaded: {len(models)}")

def get_multi_vision_summary(conf=0.4):
    """Run multiple YOLO models on a single frame and combine results.
       Returns (summary_text, total_object_count)."""
    try:
        if not cam.isOpened():
            raise RuntimeError("Camera not accessible.")
        ok, frame = cam.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture frame.")

        all_detections = {}
        total_objs = 0

        for name, model in models.items():
            try:
                res = model.predict(frame, conf=conf, verbose=False)[0]
                names = model.model.names if hasattr(model.model, "names") else model.names
                labels = [names[int(b.cls)] for b in res.boxes]
                total_objs += len(labels)
                if labels:
                    all_detections[name] = Counter(labels)
            except Exception as inner:
                print(f"❌ {name} detection error: {inner}")

        if total_objs == 0:
            return "You are currently seeing: nothing.", 0

        parts = []
        for name, objs in all_detections.items():
            parts.append(f"{name}: " + ", ".join([f"{k}Ã—{v}" for k, v in objs.items()]))
        return "You are currently seeing: " + " | ".join(parts) + ".", total_objs

    except Exception as e:
        print("❌ Vision error:", e)
        return "You are currently seeing: nothing.", 0

# Load voice
print("👄 Loading ASR + TTS models...")
whisper = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
voice   = PiperVoice.load(TTS_VOICE)

# Knowledge and instincts
print(f"📚 Connecting to Chroma (persistent) at: {CHROMA_PATH}")
client  = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(name: Optional[str] = CHROMA_COLLECTION):
    if name:
        return client.get_collection(name=name)
    cols = client.list_collections()
    if not cols:
        raise RuntimeError(f"No collections found in {CHROMA_PATH}")
    return client.get_collection(name=cols[0].name)

col = get_collection()
print(f"📖 Using collection: {col.name}")

embedder = SentenceTransformer(EMBED_MODEL)

# System prompt
def load_system_prompt_plain(path=SYSTEM_PROMPT_PATH):
    # Read the file as plain UTF-8 text (no base64).
    if not os.path.exists(path):
        print(f"❌ No system prompt found at {path}, using fallback.")
        return "You are AstraMech. Be concise and factual."
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text
    except Exception as e:
        print(f"❌ Error reading system prompt: {e}")
        return "You are AstraMech. Be concise and factual."

# Input from ears
def record_wav(path, seconds=RECORD_SECONDS):
    print(f"🔊 Recording {seconds}s...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="float32")
    sd.wait()
    sf.write(path, audio, SAMPLE_RATE)
    print("🔈 Recorded")

# Transcribe what you heard
def transcribe(path):
    print("🧠 Transcribing...")
    segments, _ = whisper.transcribe(path, beam_size=1, vad_filter=True)
    text = "".join(s.text for s in segments).strip()
    print(f"🦻🏻 Heard: {text or '[EMPTY]'}")
    return text

# Retrieve knowledge context database
def retrieve_context(query: str, k: int = TOP_K):
    print("💡 Retrieving context from Chroma...")
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=qvec, n_results=k,
                    include=["documents", "metadatas", "distances"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return list(zip(docs, metas, dists))

# Robot Intent - Prompt, basically
def compose_prompt(user_query: str, hits: List[tuple]):
    ctx_lines = []
    for i, (doc, meta, dist) in enumerate(hits, 1):
        tag = (meta or {}).get("source") or (meta or {}).get("file") or (meta or {}).get("id") or f"chunk_{i}"
        ctx_lines.append(f"[{tag}] {doc}")
    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context retrieved)"

    sys_prompt = load_system_prompt_plain()

    full_prompt = (
        f"{sys_prompt}\n\n"
        f"# Context\n{context_block}\n\n"
        f"# User said:{user_query}\n\n"
        f"# Answer the user directly and concisely below using straight sentences:\n"
    )
    return full_prompt

# Ask the language model
def ask_lmstudio(user_query: str):
    hits   = retrieve_context(user_query, k=TOP_K)
    prompt = compose_prompt(user_query, hits)
    print("📝 Asking LM Studio...")
    r = requests.post(
        LMSTUDIO_URL,
        headers={"Authorization": "Bearer lm-studio"},
        json={"model": LMSTUDIO_MODEL, "messages": [{"role": "user", "content": prompt}]},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# Speaking
def speak(text, out_path=None, play=True):
    print("👄 Speaking (streaming)...")
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

# Did we get input?
def should_query_llm(audio_active: bool, visual_active: bool, forced_trigger: bool) -> bool:
    """
    Returns True if we should call LM Studio for a response.
    - OBSERVER_MODE: only when there's signal (audio OR visual)
    - CHATTER_MODE : allow manual Enter to force a reply even without signals
    """
    if CHATTER_MODE:
        # In chatter mode, Enter can force a reply; otherwise require signal.
        return forced_trigger or audio_active or visual_active
    # Default: observer mode, only talk if there's something to talk about.
    return audio_active or visual_active

# -------- loop --------
try:
    if CHATTER_MODE:
        print("🗣 CHATTER MODE → Manual trigger only (no auto polling).")
        while True:
            wait_for_enter_blocking()   # <-- clear, visible prompt

            # audio
            wav_path, rms, _ = capture_audio_to_temp(RECORD_SECONDS)
            try:
                transcript = transcribe(wav_path)
            finally:
                try: os.remove(wav_path)
                except FileNotFoundError: pass
            audio_active = has_audio_activity(rms, transcript)

            # vision
            vision_desc, obj_count = get_multi_vision_summary()
            visual_active = (obj_count >= MIN_VISUAL_OBJECTS)
            print(f"👁 visual_active={visual_active} | objs={obj_count} | desc='{vision_desc}'")
            print(f"👂 audio_active={audio_active}")

            if not (audio_active or visual_active):
                print("🟡 Idle → nothing to talk about.")
                continue

            user_query = transcript if transcript.strip() else "What do you see in front of you"
            q_with_vision = f"{user_query}\n\n{vision_desc}"

            reply = ask_lmstudio(q_with_vision)
            print(f"🤖 LM Studio: {reply}")
            speak(reply, out_path=None, play=True)
            # loop repeats and prints the Enter prompt again

    else:
        # ------- OBSERVER MODE (auto polling every POLL_INTERVAL sec) -------
        print(f"🤖 OBSERVER MODE active → Auto-polling every {POLL_INTERVAL}s.")
        print("Press Enter anytime to trigger manually. Ctrl+C to exit.")
        while True:
            forced_trigger = wait_or_enter(POLL_INTERVAL)

            wav_path, rms, _ = capture_audio_to_temp(RECORD_SECONDS)
            try:
                transcript = transcribe(wav_path)
            finally:
                try: os.remove(wav_path)
                except FileNotFoundError: pass

            audio_active = has_audio_activity(rms, transcript)
            vision_desc, obj_count = get_multi_vision_summary()
            visual_active = (obj_count >= MIN_VISUAL_OBJECTS)
            print(f"👁 visual_active={visual_active} | objs={obj_count} | desc='{vision_desc}'")
            print(f"👂 audio_active={audio_active} | ⌨️ forced={forced_trigger}")

            if not should_query_llm(audio_active, visual_active, forced_trigger):
                print("🟡 Idle → LLM skipped (no audio/visual signals).")
                continue

            user_query = transcript if transcript.strip() else "What do you see in front of you"
            q_with_vision = f"{user_query}\n\n{vision_desc}"

            print("💬 Querying LM Studio…")
            reply = ask_lmstudio(q_with_vision)
            print(f"🤖 LM Studio: {reply}")
            speak(reply, out_path=None, play=True)
            print(f"⏳ Ready — next poll in {POLL_INTERVAL}s (or press Enter)…", flush=True)
except KeyboardInterrupt:
    print("\nBye! 👋")