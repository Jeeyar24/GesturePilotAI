
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import pyautogui
import time
from pathlib import Path

# ==========================================
# 1. PROJECT FILES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gesture_model.joblib"
HAND_PATH = BASE_DIR / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise SystemExit("Error: gesture_model.joblib is missing.")

if not HAND_PATH.exists():
    raise SystemExit("Error: hand_landmarker.task is missing.")

# Load the trained machine learning model
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

# ==========================================
# 2. GESTURE COMMANDS
# ==========================================

COMMANDS = {
    "open_palm": ("right", "NEXT SLIDE"),
    "fist": ("left", "PREVIOUS SLIDE"),
    "peace": ("b", "BLACK SCREEN"),
    "thumbs_up": ("w", "WHITE SCREEN")
}

# ==========================================
# 3. HAND LANDMARK CONNECTIONS
# ==========================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def extract_features(hand):
    """Convert 21 hand landmarks into 63 model features."""
    wrist = hand[0]

    scale = max(
        max(abs(p.x - wrist.x), abs(p.y - wrist.y))
        for p in hand
    )
    scale = max(scale, 0.001)

    features = []

    for point in hand:
        features.extend([
            (point.x - wrist.x) / scale,
            (point.y - wrist.y) / scale,
            (point.z - wrist.z) / scale
        ])

    return features


# ==========================================
# 4. STABILITY AND COOLDOWN SETTINGS
# ==========================================

CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES = 8
RESET_FRAMES = 12
ACTION_COOLDOWN = 2.0

candidate = None
candidate_count = 0

last_triggered = None
last_action_time = 0.0

# Prevent another command until the hand is removed
command_latched = False
no_hand_count = 0

status = "Show your hand"
window_name = "GesturePilot AI - PowerPoint Control"

# ==========================================
# 5. START WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Error: Could not open webcam.")

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

# Keep the preview visible when supported
try:
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_TOPMOST,
        1
    )
except cv2.error:
    pass

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

print("\nGesturePilot AI - PowerPoint Control")
print("------------------------------------")
print("Open palm  : Next slide")
print("Fist       : Previous slide")
print("Peace sign : Black screen")
print("Thumbs up  : White screen")
print("------------------------------------")
print("Hold a gesture steadily to trigger a command.")
print("Remove your hand briefly before the next command.")
print("Press Q in the camera window to quit.")
print("Ctrl+C in this terminal also stops the program.")

# ==========================================
# 6. LIVE GESTURE RECOGNITION
# ==========================================

try:
    with Vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
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
                image,
                timestamp
            )

            detected_label = None
            confidence = 0.0
            hand_present = bool(result.hand_landmarks)

            # ----------------------------------
            # 7. DETECT AND CLASSIFY THE HAND
            # ----------------------------------

            if hand_present:
                hand = result.hand_landmarks[0]

                # Draw hand connections
                for a, b in CONNECTIONS:
                    point_a = (
                        int(hand[a].x * width),
                        int(hand[a].y * height)
                    )
                    point_b = (
                        int(hand[b].x * width),
                        int(hand[b].y * height)
                    )

                    cv2.line(
                        frame,
                        point_a,
                        point_b,
                        (0, 255, 0),
                        2
                    )

                # Draw landmark points
                for point in hand:
                    cv2.circle(
                        frame,
                        (
                            int(point.x * width),
                            int(point.y * height)
                        ),
                        4,
                        (0, 0, 255),
                        -1
                    )

                # Predict using the trained Random Forest
                values = extract_features(hand)

                input_data = pd.DataFrame(
                    [values],
                    columns=FEATURES
                )

                probabilities = model.predict_proba(input_data)[0]
                best_index = probabilities.argmax()

                confidence = float(
                    probabilities[best_index]
                )

                prediction = str(
                    model.classes_[best_index]
                )

                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_label = prediction

            # ----------------------------------
            # 8. STABLE GESTURE AND LOCK LOGIC
            # ----------------------------------

            now = time.perf_counter()

            if detected_label is not None:
                # A recognized hand is visible
                no_hand_count = 0

                if detected_label == candidate:
                    candidate_count += 1
                else:
                    candidate = detected_label
                    candidate_count = 1

                status = detected_label.replace("_", " ")

                # Trigger once per hand presentation
                if (
                    candidate_count >= STABLE_FRAMES
                    and not command_latched
                    and detected_label != last_triggered
                    and now - last_action_time >= ACTION_COOLDOWN
                ):
                    key, action = COMMANDS[detected_label]

                    # Send keyboard input to active application
                    pyautogui.press(key)

                    last_action_time = now
                    last_triggered = detected_label
                    command_latched = True

                    status = action

                    print(
                        f"Gesture: {detected_label} -> {action}"
                    )

                elif command_latched:
                    status = "Command sent - remove hand"

            else:
                candidate = None
                candidate_count = 0

                if hand_present:
                    # Hand visible, but prediction is uncertain.
                    # Do not unlock while the hand remains visible.
                    no_hand_count = 0
                    status = "Gesture unclear - hold steady"

                else:
                    # Unlock only when the camera sees no hand
                    no_hand_count += 1

                    if command_latched:
                        status = "Remove hand to re-arm"
                    else:
                        status = "Show your hand"

                    if no_hand_count >= RESET_FRAMES:
                        command_latched = False
                        last_triggered = None

            # ----------------------------------
            # 9. DISPLAY STATUS
            # ----------------------------------

            cv2.putText(
                frame,
                f"Gesture: {status}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.0%}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Remove hand before next command",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(window_name, frame)

            # Press Q while the camera window is focused
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("\nStopping GesturePilot AI...")

finally:
    camera.release()
    cv2.destroyAllWindows()

print("GesturePilot AI stopped.")
