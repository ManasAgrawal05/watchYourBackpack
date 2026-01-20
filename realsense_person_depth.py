import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not found. Install with: pip install pyrealsense2")
    sys.exit(1)

from ultralytics import YOLO


PERSON_CLASS_ID = 0  # COCO 'person' in Ultralytics/YOLO APIs


def parse_args():
    p = argparse.ArgumentParser(description="YOLO11 person detection + RealSense aligned depth at centroid.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)

    # Choose a YOLO11 checkpoint; 'n' is fastest, 's/m/l/x' are heavier/more accurate.
    p.add_argument("--weights", type=str, default="yolo11n.pt")

    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--depth_window", type=int, default=7, help="Odd window for median depth around centroid.")
    p.add_argument("--warmup_frames", type=int, default=15)

    p.add_argument("--visualize", action="store_true")
    p.add_argument("--one_shot", action="store_true")
    return p.parse_args()


def robust_depth_meters(depth_frame: rs.depth_frame, cx: int, cy: int, window: int) -> float:
    if window <= 1:
        return float(depth_frame.get_distance(cx, cy))

    if window % 2 == 0:
        window += 1
    half = window // 2

    w = depth_frame.get_width()
    h = depth_frame.get_height()

    xs = range(max(0, cx - half), min(w - 1, cx + half) + 1)
    ys = range(max(0, cy - half), min(h - 1, cy + half) + 1)

    vals = []
    for y in ys:
        for x in xs:
            d = float(depth_frame.get_distance(x, y))
            if d > 0:
                vals.append(d)

    if not vals:
        return 0.0
    return float(np.median(np.array(vals, dtype=np.float32)))


def main():
    args = parse_args()

    # Load detector (downloads weights automatically if not present)
    model = YOLO(args.weights)

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"ERROR starting RealSense pipeline: {e}")
        sys.exit(1)

    align = rs.align(rs.stream.color)

    # Warmup for exposure stabilization
    for _ in range(max(0, args.warmup_frames)):
        pipeline.wait_for_frames()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())

            # YOLO expects RGB by default, but Ultralytics will handle numpy BGR reasonably.
            # For clarity, convert to RGB:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Detect only persons (classes=[0]) for speed/cleaner outputs
            results = model.predict(
                source=rgb,
                conf=args.conf,
                classes=[PERSON_CLASS_ID],
                verbose=False
            )

            r0 = results[0]
            boxes = r0.boxes  # Ultralytics Boxes

            out_lines = []
            vis = bgr.copy()

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes, start=1):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy.tolist()
                    conf = float(box.conf[0].cpu().numpy())

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    d_m = robust_depth_meters(depth_frame, cx, cy, args.depth_window)

                    out_lines.append(
                        f"person#{i:02d} conf={conf:.2f} centroid=({cx},{cy}) depth={d_m:.3f} m bbox=({x1},{y1})-({x2},{y2})"
                    )

                    if args.visualize:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(
                            vis,
                            f"{d_m:.2f} m",
                            (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

            print("-" * 72)
            if out_lines:
                for line in out_lines:
                    print(line)
            else:
                print("No persons detected above threshold.")

            if args.visualize:
                cv2.imshow("YOLO11 + RealSense Depth", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if args.one_shot:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
