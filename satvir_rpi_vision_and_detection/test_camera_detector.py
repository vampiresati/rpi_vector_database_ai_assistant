"""Manual camera/model smoke test.

Examples:
    uv run python satvir_rpi_vision_and_detection/test_camera_detector.py --once --no-window
    uv run python satvir_rpi_vision_and_detection/test_camera_detector.py
"""
import argparse
import sys

import cv2

import config
from camera_detector import Detector


def rotate_and_crop(frame):
    if config.CAMERA_ROTATE == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    left, top, right, bottom = config.FRAME_CROP
    if any(config.FRAME_CROP):
        h, w = frame.shape[:2]
        x1, y1 = int(w * left), int(h * top)
        x2, y2 = int(w * (1 - right)), int(h * (1 - bottom))
        frame = frame[y1:y2, x1:x2]
    return frame


def open_camera(index):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {index}")
    return camera


def draw_tracks(frame, tracks):
    for track in tracks:
        print(
            "Track ID:",
            track.tid,
            "Box:",
            [round(x, 1) for x in track.box],
            "Quality:",
            round(track.crop_q, 3),
            "Salience:",
            round(track.salience, 3),
        )
        x1, y1, x2, y2 = map(int, track.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Object {track.tid}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--frames", type=int, default=5,
                        help="Frames to process in --once mode.")
    args = parser.parse_args()

    detector = Detector()
    print(f"Using detector device={detector.device} quantize={detector.quantize}")
    detector.warm()

    camera = open_camera(args.camera)
    try:
        frame_count = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read camera frame")
            frame = rotate_and_crop(frame)
            tracks = detector.process(frame)
            frame_count += 1
            print(f"Frame {frame_count}: {len(tracks)} stable track(s)")
            draw_tracks(frame, tracks)

            if not args.no_window:
                cv2.imshow("Robot Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.once and frame_count >= args.frames:
                break
    finally:
        camera.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"camera detector test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
