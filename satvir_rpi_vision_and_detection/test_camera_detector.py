import cv2

from camera_detector import Detector


detector = Detector()

detector.warm()

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera")
        break

    tracks = detector.process(frame)

    for track in tracks:
        print(
            "Track ID:",
            track.tid,
            "Box:",
            track.box,
            "Quality:",
            track.crop_q,
            "Salience:",
            track.salience,
        )

        x1, y1, x2, y2 = map(int, track.box)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Object {track.tid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Robot Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
