
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import time
from pathlib import Path

# Locate the trained model and hand detector files
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gesture_model.joblib"
HAND_PATH = BASE_DIR / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise SystemExit("Run train_model.py first.")

if not HAND_PATH.exists():
    raise SystemExit("hand_landmarker.task is missing.")

# Load our trained ML model
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

# Hand landmark connections for drawing
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def extract_features(hand):
    wrist = hand[0]

    scale = max(
        max(abs(p.x - wrist.x), abs(p.y - wrist.y))
        for p in hand
    )
    scale = max(scale, 0.001)

    features = []
    for p in hand:
        features.extend([
            (p.x - wrist.x) / scale,
            (p.y - wrist.y) / scale,
            (p.z - wrist.z) / scale
        ])

    return features

# Set up the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam.")

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision

options = Vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_PATH)
    ),
    running_mode=Vision.RunningMode.VIDEO,
    num_hands=1
)

start_time = time.perf_counter()
last_timestamp = -1

print("Live gesture recognition started.")
print("Press Q to quit.")

try:
    with Vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (time.perf_counter() - start_time) * 1000
            )
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = detector.detect_for_video(
                image, timestamp
            )

            label = "No hand detected"
            confidence = 0.0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Draw the hand landmarks
                for a, b in connections:
                    p1 = (
                        int(hand[a].x * width),
                        int(hand[a].y * height)
                    )
                    p2 = (
                        int(hand[b].x * width),
                        int(hand[b].y * height)
                    )
                    cv2.line(
                        frame, p1, p2, (0, 255, 0), 2
                    )

                for p in hand:
                    cv2.circle(
                        frame,
                        (int(p.x * width), int(p.y * height)),
                        4, (0, 0, 255), -1
                    )

                # Predict the gesture using our ML model
                values = extract_features(hand)
                input_data = pd.DataFrame(
                    [values], columns=FEATURES
                )

                probabilities = model.predict_proba(input_data)[0]
                best_index = probabilities.argmax()
                confidence = float(probabilities[best_index])
                prediction = model.classes_[best_index]

                if confidence >= 0.60:
                    label = str(prediction).replace("_", " ")
                else:
                    label = "Uncertain"

            display_text = f"{label} ({confidence:.0%})"

            cv2.putText(
                frame, display_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2
            )

            cv2.imshow("GesturePilot AI - Live Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Live recognition stopped.")
