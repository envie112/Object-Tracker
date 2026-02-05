import cv2
import time

video = cv2.VideoCapture(0)

ret, frame = video.read()
if not ret:
    print("Failed to read video")
    exit()

bbox = cv2.selectROI("Select Object", frame, False)
cv2.destroyWindow("Select Object")

tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

prev_time = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Object Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        bbox = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)

video.release()
cv2.destroyAllWindows()
