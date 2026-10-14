
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Webcam started! Press Q to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("GesturePilot AI - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
