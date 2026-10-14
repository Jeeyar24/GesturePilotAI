
import cv2
import mediapipe as mp
import csv
import time
from pathlib import Path

# Files are stored in the project folder
MODEL = Path(__file__).with_name("hand_landmarker.task")
DATA = Path(__file__).with_name("gesture_data.csv")

GESTURES = {
    "1": "open_palm",
    "2": "fist",
    "3": "peace",
    "4": "thumbs_up"
}

if not MODEL.exists():
    print("Missing hand_landmarker.task model file.")
    raise SystemExit(1)

# Create CSV file and headings if needed
if not DATA.exists():
    with DATA.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["label"] + [f"f{i}" for i in range(63)]
        )

BaseOptions = mp.tasks.BaseOptions
Vision = mp.tasks.vision
HandLandmarker = Vision.HandLandmarker
Options = Vision.HandLandmarkerOptions
RunningMode = Vision.RunningMode

options = Options(
    base_options=BaseOptions(model_asset_path=str(MODEL)),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def extract_features(hand):
    # Make coordinates relative to the wrist
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


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    raise SystemExit(1)

current_label = None
current_features = None
saved = 0
start = time.perf_counter()
last_timestamp = -1

print("1=Open palm, 2=Fist, 3=Peace, 4=Thumbs up")
print("Choose a label, then press SPACE to save a sample.")
print("Press Q to quit.")

try:
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (time.perf_counter() - start) * 1000
            )
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = detector.detect_for_video(image, timestamp)
            current_features = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                current_features = extract_features(hand)

                for a, b in connections:
                    p1 = (int(hand[a].x * width),
                          int(hand[a].y * height))
                    p2 = (int(hand[b].x * width),
                          int(hand[b].y * height))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

                for p in hand:
                    cv2.circle(
                        frame,
                        (int(p.x * width), int(p.y * height)),
                        4, (0, 0, 255), -1
                    )

            label_text = current_label or "Choose 1, 2, 3 or 4"
            cv2.putText(
                frame, f"Gesture: {label_text}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                frame, f"Samples saved: {saved}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, "SPACE=Save sample | Q=Quit", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            cv2.imshow("GesturePilot - Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
                current_label = GESTURES[chr(key)]
                print("Selected:", current_label)

            elif key == ord(" ") and current_label and current_features:
                with DATA.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [current_label] + current_features
                    )

                saved += 1
                print(f"Saved {current_label}: sample {saved}")

            elif key == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Collection finished. Samples added this session: {saved}")
print(f"Dataset: {DATA}")
