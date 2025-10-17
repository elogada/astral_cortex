# vision-test
# Reads detections from vision-latest.json produced by vision-check.py
# and returns a friendly summary string.

import json
from pathlib import Path
from datetime import datetime

LATEST_FILE = Path("vision-latest.json")

def read_latest():
    if not LATEST_FILE.exists():
        return None
    try:
        return json.loads(LATEST_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Can't read {LATEST_FILE}: {e}")
        return None

def summarize_detection(data, max_objs=3):
    if not data or "detections" not in data or not data["detections"]:
        return "No recent detections."

    dets = data["detections"][:max_objs]
    ts = datetime.fromtimestamp(data["ts"]).strftime("%H:%M:%S")

    parts = []
    for d in dets:
        obj = d["object"]
        orient = d["orientation"]
        ang = d["angle_deg"]
        conf = d["confidence"] * 100
        src = d.get("source_model", "?")
        parts.append(f"{obj} ({orient}, {ang:+.1f}°, {conf:.1f}% via )")

    joined = "; ".join(parts)
    return f"As of {ts}, I can see {joined}."

if __name__ == "__main__":
    info = read_latest()
    print(summarize_detection(info))