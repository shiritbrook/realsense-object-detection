"""
RealSense object detection with distance
----------------------------------------
Detects objects in the RealSense color stream with a YOLO model and uses the
aligned depth stream to show how far away each object is (in meters), plus its
3D position relative to the camera.

Based on the official librealsense Python examples:
  * align-depth2color.py  - streaming, depth-to-color alignment, depth scale
  * read_bag_example.py   - playing back a recorded .bag file instead of a camera
  * distance_to_object    - measuring distance inside a detected bounding box

Requirements (Python 3.9+):
    pip install pyrealsense2 ultralytics opencv-python numpy

Usage:
    python realsense_object_detection.py                     # live camera
    python realsense_object_detection.py --bag recording.bag # recorded file
    python realsense_object_detection.py --classes person cup --conf 0.5
    python realsense_object_detection.py --save output.mp4   # also record video

Keys: q or Esc = quit, d = show/hide depth view, s = save a snapshot
"""

import argparse
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Missing package: run  pip install ultralytics")

WIDTH, HEIGHT, FPS = 640, 480, 30


# ------------------------------------------------------------------ camera --
def start_pipeline(bag_file=None):
    """Start streaming color + depth from a camera or a .bag file.
    Returns (pipeline, align, depth_scale)."""
    pipeline = rs.pipeline()
    config = rs.config()

    if bag_file:
        # Same approach as read_bag_example.py: play back a recording
        rs.config.enable_device_from_file(config, bag_file, repeat_playback=True)
        config.enable_stream(rs.stream.depth)
        config.enable_stream(rs.stream.color)
    else:
        # Same checks as align-depth2color.py: a camera with an RGB sensor
        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            sys.exit("No RealSense camera found. Connect one, or use --bag <file>.")
        wrapper = rs.pipeline_wrapper(pipeline)
        device = config.resolve(wrapper).get_device()
        if not any(s.get_info(rs.camera_info.name) == "RGB Camera" for s in device.sensors):
            sys.exit("This demo needs a depth camera with a color sensor (e.g. D435, D455).")
        print("Camera:", device.get_info(rs.camera_info.name))
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale} meters per unit")

    # Align depth to color so a pixel in a color bounding box maps to its depth
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale


def get_frames(pipeline, align):
    """Wait for the next aligned color + depth pair. Returns (color, depth_frame)
    or (None, None) if a frame was dropped."""
    frames = align.process(pipeline.wait_for_frames())
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None
    color = np.asanyarray(color_frame.get_data())
    # .bag recordings are often stored as RGB; OpenCV and YOLO expect BGR
    if color_frame.get_profile().format() == rs.format.rgb8:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    return color, depth_frame


# ---------------------------------------------------------------- distance --
def object_distance(depth_image, box, depth_scale):
    """Distance in meters to the object inside `box` (x1, y1, x2, y2).

    Uses the median of the valid depth values in the central half of the box,
    so background pixels at the box edges and missing depth (0) don't skew it.
    Returns None if there is no valid depth in that area."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + w * 0.25), int(x2 - w * 0.25)
    cy1, cy2 = int(y1 + h * 0.25), int(y2 - h * 0.25)
    region = depth_image[max(cy1, 0):max(cy2, cy1 + 1), max(cx1, 0):max(cx2, cx1 + 1)]
    valid = region[region > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * depth_scale


def object_position(depth_frame, box, distance):
    """3D point (x, y, z) in meters of the box center, relative to the camera:
    x = right, y = down, z = forward."""
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance)


# ----------------------------------------------------------------- drawing --
def color_for(class_id):
    rng = np.random.default_rng(class_id)  # stable color per class
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_detection(img, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1, th + 8)
    cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
    cv2.putText(img, label, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA)


def detect(model, color, depth_frame, depth_scale, conf, class_ids):
    """Run YOLO on the color image and attach a distance to each detection.
    Returns a list of dicts: name, conf, box, distance, position."""
    depth_image = np.asanyarray(depth_frame.get_data())
    result = model(color, conf=conf, classes=class_ids, verbose=False)[0]
    detections = []
    for b in result.boxes:
        box = tuple(int(v) for v in b.xyxy[0].tolist())
        cls = int(b.cls[0])
        dist = object_distance(depth_image, box, depth_scale)
        pos = object_position(depth_frame, box, dist) if dist else None
        detections.append({"name": model.names[cls], "cls": cls, "conf": float(b.conf[0]),
                           "box": box, "distance": dist, "position": pos})
    return detections


# -------------------------------------------------------------------- main --
def parse_args():
    p = argparse.ArgumentParser(description="RealSense object detection with distance")
    p.add_argument("--bag", help="play a recorded .bag file instead of a live camera")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO model (downloaded automatically on first run), default yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4, help="minimum confidence, default 0.4")
    p.add_argument("--classes", nargs="+", help="only detect these classes, e.g. person cup chair")
    p.add_argument("--save", help="also record the annotated video to this .mp4 file")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model {args.model} ...")
    model = YOLO(args.model)
    class_ids = None
    if args.classes:
        lookup = {name.lower(): i for i, name in model.names.items()}
        unknown = [c for c in args.classes if c.lower() not in lookup]
        if unknown:
            sys.exit(f"Unknown class(es): {', '.join(unknown)}. "
                     f"Available: {', '.join(model.names.values())}")
        class_ids = [lookup[c.lower()] for c in args.classes]

    pipeline, align, depth_scale = start_pipeline(args.bag)
    colorizer = rs.colorizer()
    writer, show_depth, last_print = None, True, 0.0
    window = "RealSense Object Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        prev = time.time()
        while True:
            color, depth_frame = get_frames(pipeline, align)
            if color is None:
                continue

            detections = detect(model, color, depth_frame, depth_scale, args.conf, class_ids)

            view = color.copy()
            for d in detections:
                dist = f"{d['distance']:.2f} m" if d["distance"] else "no depth"
                draw_detection(view, d["box"], f"{d['name']} {d['conf']:.0%}  {dist}",
                               color_for(d["cls"]))

            now = time.time()
            fps, prev = 1.0 / max(now - prev, 1e-6), now
            cv2.putText(view, f"{fps:.1f} FPS  |  {len(detections)} objects", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if show_depth:
                depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                depth_vis = cv2.resize(depth_vis, (view.shape[1], view.shape[0]))
                if depth_vis.shape[2] == 3:
                    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
                for d in detections:
                    x1, y1, x2, y2 = d["box"]
                    cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
                output = np.hstack((view, depth_vis))
            else:
                output = view

            # Print a short text summary about once a second
            if now - last_print > 1.0 and detections:
                last_print = now
                for d in detections:
                    if d["position"]:
                        x, y, z = d["position"]
                        print(f"{d['name']:<12} {d['distance']:.2f} m   "
                              f"(x={x:+.2f}, y={y:+.2f}, z={z:.2f})")

            if args.save:
                if writer is None:
                    rec_size = (output.shape[1], output.shape[0])
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                             rec_size)
                # keep the recording a fixed size even if the depth view is toggled
                frame = output if (output.shape[1], output.shape[0]) == rec_size \
                    else cv2.resize(output, rec_size)
                writer.write(frame)

            cv2.imshow(window, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_depth = not show_depth
            if key == ord("s"):
                name = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, output)
                print("Saved", name)
    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
