
# vision-check.py
# Standalone YOLO + OpenCV "eye" script for AstraMech
#
# New:
#   - Multi-model support. Load general + specialist models and merge detections.
#   - Flags:
#       --use-all-models     -> load ALL valid models from vision-list.ini (default was first-only)
#       --add-model PATH     -> add extra model(s); repeat flag to add multiple
#       --topk N             -> cap number of merged detections per frame (default 1)
#       --lines-per-detection-> print one JSON per detection (stdout) instead of one array
#
# Examples:
#   python vision-check.py --no-preview --latest-file vision-latest.json --topk 3 --use-all-models
#   python vision-check.py --no-preview --topk 3 --add-model yolov8-pest-detection.pt
#
# Requirements: pip install ultralytics opencv-python
import cv2
import sys, time, json, os, argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    print("Fatal: ultralytics not installed or failed to import. Try: pip install ultralytics")
    sys.exit(1)

INI_NAME = "vision-list.ini"
DEFAULT_MODEL = "yolov8n.pt"

def read_model_candidates(ini_path: Path):
    if not ini_path.exists():
        return []
    out = []
    for line in ini_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        out.append(s)
    return out

def try_load_model(name):
    try:
        m = YOLO(name)
        import numpy as np
        _ = m.predict(np.zeros((1,1,3), dtype="uint8"), verbose=False, imgsz=32)
        return m
    except Exception:
        return None

def load_models(use_all=False, add_models=None):
    ini_path = Path(__file__).with_name(INI_NAME)
    candidates = read_model_candidates(ini_path)
    loaded = []
    names = []

    if use_all:
        for cand in candidates:
            m = try_load_model(cand)
            if m is not None:
                loaded.append(m); names.append(cand)
    else:
        # original behavior: first working
        for cand in candidates:
            m = try_load_model(cand)
            if m is not None:
                loaded.append(m); names.append(cand)
                break

    # add any explicitly passed models
    for extra in (add_models or []):
        m = try_load_model(extra)
        if m is not None:
            loaded.append(m); names.append(extra)

    if not loaded:
        # fallback to default
        m = try_load_model(DEFAULT_MODEL)
        if m is None:
            print(json.dumps({"event":"fatal","error":"no_model_loaded"}))
            sys.exit(2)
        loaded = [m]
        names = [DEFAULT_MODEL]
        print(json.dumps({"event":"model_fallback","note":"could_not_load_from_ini_or_args","fallback":DEFAULT_MODEL}))
    else:
        src = "ini_all" if use_all else "ini_first"
        print(json.dumps({"event":"models_loaded","source":src,"models":names}))
    return loaded, names

def compute_orientation_deg(x_center, img_width, hfov_deg=60.0):
    deg_per_px = hfov_deg / float(max(1, img_width))
    offset_px = x_center - (img_width / 2.0)
    return offset_px * deg_per_px

def orientation_label(angle_deg, dead_zone=3.0):
    if abs(angle_deg) <= dead_zone:
        return "center"
    return "left" if angle_deg < 0 else "right"

def atomic_write(path: Path, data: str):
    tmp = Path(f"{path}.{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)

def det_json(obj_name, x1, y1, x2, y2, w, h, angle_deg, conf, source_model):
    return {
        "object": obj_name,
        "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
        "center": {"x": int((x1+x2)/2.0), "y": int((y1+y2)/2.0)},
        "orientation": "left" if angle_deg < -3 else ("right" if angle_deg > 3 else "center"),
        "angle_deg": round(float(angle_deg), 2),
        "confidence": round(float(conf), 4),
        "source_model": source_model
    }

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oneshot", action="store_true", help="Capture a single frame and exit")
    ap.add_argument("--no-preview", action="store_true", help="Disable OpenCV window")
    ap.add_argument("--latest-file", type=Path, help="Continuously overwrite this JSON with latest detection(s)")
    ap.add_argument("--out", type=Path, help="Write JSON output here in --oneshot mode")
    ap.add_argument("--hfov", type=float, default=60.0, help="Horizontal field-of-view in degrees")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--print-rate", type=float, default=0.0, help="Min seconds between stdout prints (0 = unthrottled)")
    ap.add_argument("--topk", type=int, default=1, help="Max detections per frame to output")
    ap.add_argument("--lines-per-detection", action="store_true", help="Print one JSON line per detection")
    ap.add_argument("--use-all-models", action="store_true", help="Load all valid models from vision-list.ini")
    ap.add_argument("--add-model", action="append", default=[], help="Add an extra model (repeatable)")
    return ap.parse_args()

# IoU and NMS for cross-model merge
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def nms_merge(dets, iou_thresh=0.55):
    # dets: list of (conf, name, x1,y1,x2,y2, angle, src_model)
    dets = sorted(dets, key=lambda x: x[0], reverse=True)
    picked = []
    for d in dets:
        keep = True
        for p in picked:
            if iou((d[2],d[3],d[4],d[5]), (p[2],p[3],p[4],p[5])) >= iou_thresh:
                keep = False
                break
        if keep:
            picked.append(d)
    return picked

def main():
    args = parse_args()
    if args.topk < 1:
        args.topk = 1

    models, model_names = load_models(use_all=args.use_all_models, add_models=args.add_model)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(json.dumps({"event":"fatal","error":"cannot_open_webcam"}))
        sys.exit(3)

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(json.dumps({"event":"warn","warning":"read_frame_failed"}))
                break

            h, w = frame.shape[:2]
            merged = []

            # run each model and collect detections
            for m, mname in zip(models, model_names):
                results = m.predict(source=frame, conf=args.conf, iou=0.45, verbose=False)
                for r in results or []:
                    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
                        continue
                    names_map = r.names if hasattr(r, "names") else {}
                    for b in r.boxes:
                        try:
                            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                            cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                            name = names_map.get(cls_id, f"id:{cls_id}")
                            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                            cx = (x1 + x2) / 2.0
                            angle = compute_orientation_deg(cx, w, args.hfov)
                            merged.append((conf, name, x1, y1, x2, y2, angle, mname))
                        except Exception:
                            continue

            # merge with simple NMS across models
            merged = nms_merge(merged, iou_thresh=0.55)
            # top-k
            merged = merged[:args.topk]

            ts = time.time()
            out = {
                "models": model_names,
                "ts": ts,
                "frame_size": {"w": int(w), "h": int(h)},
                "detections": [
                    det_json(name, x1, y1, x2, y2, w, h, angle, conf, src)
                    for (conf, name, x1, y1, x2, y2, angle, src) in merged
                ]
            }
            out["best"] = out["detections"][0] if out["detections"] else None

            # latest-file
            if args.latest_file:
                try:
                    atomic_write(args.latest_file, json.dumps(out))
                except Exception:
                    pass

            # printing control
            now = ts
            do_print = (args.print_rate <= 0) or (now - last_print >= args.print_rate)
            if do_print:
                if args.lines_per_detection:
                    if out["detections"]:
                        for d in out["detections"]:
                            line = json.dumps({"ts": ts, **d})
                            print(line)
                    else:
                        print(json.dumps({"ts": ts, "detections": []}))
                else:
                    print(json.dumps(out))
                last_print = now

            # overlay
            if not args.no_preview and out["detections"]:
                for d in out["detections"]:
                    x1, y1, x2, y2 = d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]
                    try:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        s = f"{d['object']} {d['orientation']} ({d['angle_deg']:+.1f}°) {d['confidence']*100:.1f}% [{d['source_model']}]"
                        y_text = max(25, int(0.03*h))
                        cv2.putText(frame, s, (x1, max(y1-8, y_text)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                    except Exception:
                        pass

            if not args.no_preview:
                cv2.imshow("vision-check", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.oneshot:
                if args.out:
                    try:
                        atomic_write(args.out, json.dumps(out))
                    except Exception:
                        pass
                break

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
