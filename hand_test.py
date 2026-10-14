
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Find the model file in this project folder
model_path = Path(__file__).with_name("hand_landmarker.task")

if not model_path.exists() or model_path.stat().st_size == 0:
    print("Model file is missing or empty.")
    print("Download hand_landmarker.task into this folder.")
    raise SystemExit(1)

# Configure the modern MediaPipe Hand Landmarker API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# Connections between the 21 hand landmarks
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    raise SystemExit(1)

start_time = time.perf_counter()
last_timestamp = -1

try:
    with HandLandmarker.create_from_options(options) as detector:
        print("Hand detection started. Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # MediaPipe expects RGB images
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # VIDEO mode needs an increasing timestamp
            timestamp = int(
                (time.perf_counter() - start_time) * 1000
            )
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = detector.detect_for_video(
                mp_image, timestamp
            )

            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    # Draw connections
                    for a, b in connections:
                        point_a = (
                            int(hand[a].x * width),
                            int(hand[a].y * height)
                        )
                        point_b = (
                            int(hand[b].x * width),
                            int(hand[b].y * height)
                        )
                        cv2.line(
                            frame, point_a, point_b,
                            (0, 255, 0), 2
                        )

                    # Draw landmark points
                    for point in hand:
                        x = int(point.x * width)
                        y = int(point.y * height)
                        cv2.circle(
                            frame, (x, y), 5,
                            (0, 0, 255), -1
                        )

                message = "HAND DETECTED"
                color = (0, 255, 0)
            else:
                message = "Show your hand"
                color = (0, 255, 255)

            cv2.putText(
                frame, message, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2
            )

            cv2.imshow("GesturePilot AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Program closed.")
