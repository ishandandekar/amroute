import argparse
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

WINDOW = "Ambulance Detection"


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time ambulance detection")
    parser.add_argument("--source", default="0", help="webcam index, video file, or RTSP URL")
    parser.add_argument(
        "--model", default="./best_YOLO_ambulance_detect.pt", help="path to YOLO weights"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--output", default=None, help="save annotated stream to this mp4 file")
    parser.add_argument(
        "--cooldown", type=float, default=2.0, help="seconds between detection alerts"
    )
    return parser.parse_args()


def open_source(source_arg):
    source = int(source_arg) if source_arg.isdigit() else source_arg
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source_arg}")
    return cap


def alert_detections(result, now, last_alert, cooldown):
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names.get(cls_id, str(cls_id))
        if now - last_alert >= cooldown:
            print(f"[{datetime.now():%H:%M:%S}] {label} detected (conf={conf:.2f})")
            last_alert = now
    return last_alert


def main():
    args = parse_args()
    model = YOLO(args.model, verbose=False)
    cap = open_source(args.source)

    writer = None
    fps = 0.0
    last_alert = 0.0
    prev_t = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps
            prev_t = now

            annotated = result.plot()

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            last_alert = alert_detections(result, now, last_alert, args.cooldown)

            if writer is None and args.output:
                h, w = annotated.shape[:2]
                src_fps = cap.get(cv2.CAP_PROP_FPS)
                out_fps = src_fps if src_fps > 1 else (fps if fps > 0 else 30.0)
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

            if writer is not None:
                writer.write(annotated)

            cv2.imshow(WINDOW, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
