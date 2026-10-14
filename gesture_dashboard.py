
import ctypes
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import cv2
import mediapipe as mp
import joblib
import pandas as pd
import pyautogui
from PIL import Image, ImageTk


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "gesture_model.joblib"
HAND_PATH = BASE_DIR / "hand_landmarker.task"

COMMANDS = {
    "open_palm": ("right", "NEXT SLIDE"),
    "fist": ("left", "PREVIOUS SLIDE"),
    "peace": ("b", "BLACK SCREEN"),
    "thumbs_up": ("w", "WHITE SCREEN")
}

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES = 8
RESET_FRAMES = 12
ACTION_COOLDOWN = 2.0


def extract_features(hand):
    """Convert the 21 hand landmarks into 63 model features."""
    wrist = hand[0]

    scale = max(
        max(abs(point.x - wrist.x), abs(point.y - wrist.y))
        for point in hand
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


def get_active_window_title():
    """Return the title of the currently active Windows window."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value
    except Exception:
        return ""


# ==================================================
# DASHBOARD
# ==================================================

class GesturePilotDashboard:

    def __init__(self, root):
        self.root = root
        self.root.title("GesturePilot AI | Control Center")
        self.root.geometry("1120x760")
        self.root.minsize(960, 660)
        self.root.configure(bg="#0b1120")

        # Thread-safe communication between camera and GUI
        self.stop_event = threading.Event()
        self.commands_enabled = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.event_queue = queue.Queue()

        self.worker = None
        self.closing = False

        # Dashboard values
        self.status_var = tk.StringVar(value="Camera stopped")
        self.gesture_var = tk.StringVar(value="Waiting")
        self.confidence_var = tk.StringVar(value="0%")
        self.last_action_var = tk.StringVar(value="No actions yet")
        self.control_var = tk.BooleanVar(value=False)

        self.configure_styles()
        self.build_interface()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.after(50, self.refresh_dashboard)

    # --------------------------------------------------
    # STYLING
    # --------------------------------------------------

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TFrame",
            background="#0b1120"
        )

        style.configure(
            "Card.TFrame",
            background="#151f33"
        )

        style.configure(
            "Title.TLabel",
            background="#0b1120",
            foreground="#f1f5f9",
            font=("Segoe UI", 23, "bold")
        )

        style.configure(
            "Subtitle.TLabel",
            background="#0b1120",
            foreground="#94a3b8",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Section.TLabel",
            background="#151f33",
            foreground="#e2e8f0",
            font=("Segoe UI", 13, "bold")
        )

        style.configure(
            "Muted.TLabel",
            background="#151f33",
            foreground="#94a3b8",
            font=("Segoe UI", 9)
        )

        style.configure(
            "Value.TLabel",
            background="#151f33",
            foreground="#38bdf8",
            font=("Segoe UI", 17, "bold")
        )

        style.configure(
            "TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 8)
        )

        style.configure(
            "Accent.TButton",
            background="#2563eb",
            foreground="#ffffff"
        )

        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8")]
        )

        style.configure(
            "Stop.TButton",
            background="#991b1b",
            foreground="#ffffff"
        )

        style.map(
            "Stop.TButton",
            background=[("active", "#7f1d1d")]
        )

        style.configure(
            "TCheckbutton",
            background="#151f33",
            foreground="#e2e8f0",
            font=("Segoe UI", 10)
        )

    # --------------------------------------------------
    # BUILD THE USER INTERFACE
    # --------------------------------------------------

    def build_interface(self):

        # Header
        header = ttk.Frame(self.root, padding=(20, 18, 20, 14))
        header.pack(fill="x")

        title_area = ttk.Frame(header)
        title_area.pack(side="left", fill="x", expand=True)

        ttk.Label(
            title_area,
            text="GesturePilot AI",
            style="Title.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            title_area,
            text="Real-Time Hand Gesture Recognition & Control",
            style="Subtitle.TLabel"
        ).pack(anchor="w", pady=(3, 0))

        ttk.Label(
            header,
            textvariable=self.status_var,
            foreground="#38bdf8",
            background="#0b1120",
            font=("Segoe UI", 10, "bold")
        ).pack(side="right", padx=(10, 0))

        # Main content
        body = ttk.Frame(self.root, padding=(20, 0, 20, 14))
        body.pack(fill="both", expand=True)

        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # LEFT: Camera preview
        camera_card = ttk.Frame(
            body,
            style="Card.TFrame",
            padding=14
        )
        camera_card.grid(
            row=0, column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        ttk.Label(
            camera_card,
            text="LIVE CAMERA PREVIEW",
            style="Section.TLabel"
        ).pack(anchor="w", pady=(0, 10))

        self.preview = tk.Label(
            camera_card,
            text="Camera is stopped\n\nClick Start Camera to begin",
            bg="#050914",
            fg="#cbd5e1",
            font=("Segoe UI", 13),
            width=68,
            height=20,
            relief="flat"
        )
        self.preview.pack(
            fill="both",
            expand=True
        )

        ttk.Label(
            camera_card,
            text="Green lines show tracked hand landmarks.",
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(10, 0))

        # RIGHT: Status and controls
        right_card = ttk.Frame(
            body,
            style="Card.TFrame",
            padding=16
        )
        right_card.grid(
            row=0, column=1,
            sticky="nsew"
        )

        ttk.Label(
            right_card,
            text="RECOGNITION STATUS",
            style="Section.TLabel"
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(
            right_card,
            text="Detected gesture",
            style="Muted.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            right_card,
            textvariable=self.gesture_var,
            style="Value.TLabel"
        ).pack(anchor="w", pady=(2, 10))

        ttk.Label(
            right_card,
            text="Model confidence",
            style="Muted.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            right_card,
            textvariable=self.confidence_var,
            style="Value.TLabel"
        ).pack(anchor="w", pady=(2, 10))

        ttk.Label(
            right_card,
            text="Last action",
            style="Muted.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            right_card,
            textvariable=self.last_action_var,
            foreground="#4ade80",
            background="#151f33",
            font=("Segoe UI", 10, "bold"),
            wraplength=300
        ).pack(anchor="w", pady=(3, 16))

        ttk.Separator(right_card).pack(fill="x", pady=8)

        ttk.Label(
            right_card,
            text="CAMERA CONTROLS",
            style="Section.TLabel"
        ).pack(anchor="w", pady=(4, 10))

        buttons = ttk.Frame(
            right_card,
            style="Card.TFrame"
        )
        buttons.pack(fill="x")

        self.start_button = ttk.Button(
            buttons,
            text="▶  Start Camera",
            style="Accent.TButton",
            command=self.start_camera
        )
        self.start_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 5)
        )

        self.stop_button = ttk.Button(
            buttons,
            text="■  Stop",
            style="Stop.TButton",
            command=self.stop_camera,
            state="disabled"
        )
        self.stop_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(5, 0)
        )

        ttk.Checkbutton(
            right_card,
            text="Enable PowerPoint commands",
            variable=self.control_var,
            command=self.toggle_commands
        ).pack(anchor="w", pady=(14, 4))

        ttk.Label(
            right_card,
            text=(
                "Commands are sent only when a PowerPoint "
                "or slide-show window is active."
            ),
            style="Muted.TLabel",
            wraplength=300
        ).pack(anchor="w", pady=(0, 8))

        # Bottom: Action history
        history_card = ttk.Frame(
            self.root,
            style="Card.TFrame",
            padding=(20, 10)
        )
        history_card.pack(
            fill="x",
            padx=20,
            pady=(0, 16)
        )

        history_header = ttk.Frame(
            history_card,
            style="Card.TFrame"
        )
        history_header.pack(fill="x", pady=(0, 8))

        ttk.Label(
            history_header,
            text="ACTION HISTORY",
            style="Section.TLabel"
        ).pack(side="left")

        ttk.Button(
            history_header,
            text="Clear History",
            command=self.clear_history
        ).pack(side="right")

        history_container = ttk.Frame(
            history_card,
            style="Card.TFrame"
        )
        history_container.pack(fill="x")

        self.history_list = tk.Listbox(
            history_container,
            height=5,
            bg="#0b1120",
            fg="#cbd5e1",
            selectbackground="#1d4ed8",
            selectforeground="#ffffff",
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 9)
        )
        self.history_list.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar = ttk.Scrollbar(
            history_container,
            orient="vertical",
            command=self.history_list.yview
        )
        scrollbar.pack(side="right", fill="y")

        self.history_list.configure(
            yscrollcommand=scrollbar.set
        )

    # --------------------------------------------------
    # DASHBOARD CONTROLS
    # --------------------------------------------------

    def start_camera(self):

        if self.worker is not None and self.worker.is_alive():
            return

        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Missing model",
                "gesture_model.joblib was not found.\n"
                "Run train_model.py first."
            )
            return

        if not HAND_PATH.exists():
            messagebox.showerror(
                "Missing hand model",
                "hand_landmarker.task was not found."
            )
            return

        self.stop_event.clear()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Starting camera...")

        self.worker = threading.Thread(
            target=self.camera_worker,
            daemon=True
        )
        self.worker.start()

        self.add_history("Camera start requested")

    def stop_camera(self):

        self.stop_event.set()
        self.commands_enabled.clear()
        self.control_var.set(False)

        self.status_var.set("Stopping camera...")
        self.stop_button.configure(state="disabled")

    def toggle_commands(self):

        if self.control_var.get():
            self.commands_enabled.set()
            self.add_history("PowerPoint commands enabled")
            self.status_var.set("Commands enabled")
        else:
            self.commands_enabled.clear()
            self.add_history("PowerPoint commands disabled")
            self.status_var.set("Commands disabled")

    def clear_history(self):
        self.history_list.delete(0, tk.END)

    def add_history(self, message):

        timestamp = time.strftime("%H:%M:%S")
        self.history_list.insert(
            0, f"{timestamp}  {message}"
        )

        # Keep the newest 50 entries
        while self.history_list.size() > 50:
            self.history_list.delete(tk.END)

    # --------------------------------------------------
    # CAMERA WORKER
    # --------------------------------------------------

    def camera_worker(self):

        camera = None

        try:
            bundle = joblib.load(MODEL_PATH)
            model = bundle["model"]
            features_columns = bundle["features"]

            camera = cv2.VideoCapture(0)

            if not camera.isOpened():
                raise RuntimeError(
                    "Could not open webcam. Check camera permissions "
                    "and close any other app using the camera."
                )

            BaseOptions = mp.tasks.BaseOptions
            Vision = mp.tasks.vision

            options = Vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(HAND_PATH)
                ),
                running_mode=Vision.RunningMode.VIDEO,
                num_hands=1
            )

            candidate = None
            candidate_count = 0

            command_latched = False
            no_hand_count = 0

            last_action_time = 0.0
            lock_message = ""

            start_time = time.perf_counter()
            last_timestamp = -1

            self.event_queue.put(
                ("status", "Camera running")
            )

            with Vision.HandLandmarker.create_from_options(
                options
            ) as detector:

                while not self.stop_event.is_set():

                    success, frame = camera.read()

                    if not success:
                        raise RuntimeError(
                            "Could not read a webcam frame."
                        )

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
                    timestamp = max(
                        timestamp, last_timestamp + 1
                    )
                    last_timestamp = timestamp

                    result = detector.detect_for_video(
                        image,
                        timestamp
                    )

                    hand_present = bool(result.hand_landmarks)
                    detected_label = None
                    confidence = 0.0
                    now = time.perf_counter()

                    # Detect and draw the hand
                    if hand_present:

                        hand = result.hand_landmarks[0]

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

                        values = extract_features(hand)

                        input_data = pd.DataFrame(
                            [values],
                            columns=features_columns
                        )

                        probabilities = model.predict_proba(
                            input_data
                        )[0]

                        best_index = probabilities.argmax()
                        confidence = float(
                            probabilities[best_index]
                        )

                        prediction = str(
                            model.classes_[best_index]
                        )

                        if confidence >= CONFIDENCE_THRESHOLD:
                            detected_label = prediction

                    # Gesture classification and lock
                    if detected_label is not None:

                        no_hand_count = 0

                        if detected_label == candidate:
                            candidate_count += 1
                        else:
                            candidate = detected_label
                            candidate_count = 1

                        status = detected_label.replace("_", " ")

                        if command_latched:
                            status = lock_message

                        elif (
                            candidate_count >= STABLE_FRAMES
                            and now - last_action_time >= ACTION_COOLDOWN
                        ):

                            # Lock immediately so one hand presentation
                            # cannot repeatedly send commands.
                            command_latched = True
                            last_action_time = now

                            if not self.commands_enabled.is_set():

                                lock_message = (
                                    "Controls disabled. Enable them, "
                                    "remove your hand, and try again."
                                )

                                self.event_queue.put((
                                    "log",
                                    f"Detected {detected_label}; "
                                    "command not sent (controls disabled)"
                                ))

                            else:
                                title = get_active_window_title()
                                title_lower = title.lower()

                                is_powerpoint_active = (
                                    "powerpoint" in title_lower
                                    or "slide show" in title_lower
                                )

                                if is_powerpoint_active:

                                    key, action = COMMANDS[
                                        detected_label
                                    ]

                                    pyautogui.press(key)

                                    lock_message = (
                                        f"{action} sent. Remove hand "
                                        "before the next command."
                                    )

                                    self.event_queue.put((
                                        "action",
                                        f"{action} "
                                        f"({detected_label.replace('_', ' ')})"
                                    ))

                                else:

                                    lock_message = (
                                        "PowerPoint slide show is not "
                                        "active. Activate it, remove "
                                        "your hand, then try again."
                                    )

                                    self.event_queue.put((
                                        "log",
                                        "Command skipped: PowerPoint "
                                        "slide show was not active"
                                    ))

                            status = lock_message

                    else:
                        candidate = None
                        candidate_count = 0

                        if hand_present:
                            # Hand is visible, but classification is
                            # uncertain. Do not unlock yet.
                            no_hand_count = 0
                            status = "Hand detected; gesture uncertain"

                        else:
                            no_hand_count += 1

                            if command_latched:
                                status = lock_message
                            else:
                                status = "No hand detected"

                            # Unlock only after the hand leaves view
                            if no_hand_count >= RESET_FRAMES:
                                if command_latched:
                                    lock_message = ""
                                    status = "Ready - show a gesture"

                                command_latched = False
                                no_hand_count = 0
                                candidate = None
                                candidate_count = 0

                    # Draw status on the camera image
                    display_label = (
                        detected_label.replace("_", " ")
                        if detected_label
                        else "Uncertain / no hand"
                    )

                    cv2.putText(
                        frame,
                        f"Gesture: {display_label}",
                        (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Confidence: {confidence:.0%}",
                        (10, 63),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2
                    )

                    # Send latest frame to Tkinter.
                    # The queue retains only the newest frame.
                    display_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    packet = (
                        display_rgb,
                        display_label,
                        confidence,
                        status
                    )

                    try:
                        self.frame_queue.put_nowait(packet)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass

                        try:
                            self.frame_queue.put_nowait(packet)
                        except queue.Full:
                            pass

                    time.sleep(0.005)

        except Exception as error:
            self.event_queue.put(
                ("error", str(error))
            )

        finally:
            if camera is not None:
                camera.release()

            self.event_queue.put(
                ("stopped", "Camera stopped")
            )

    # --------------------------------------------------
    # UPDATE THE GUI ON THE MAIN THREAD
    # --------------------------------------------------

    def refresh_dashboard(self):

        if self.closing:
            return

        # Show the newest camera frame
        while True:
            try:
                packet = self.frame_queue.get_nowait()
            except queue.Empty:
                break

            rgb, label, confidence, status = packet

            picture = Image.fromarray(rgb)
            picture.thumbnail((680, 420))

            photo = ImageTk.PhotoImage(picture)

            self.preview.configure(
                image=photo,
                text=""
            )
            self.preview.image = photo

            self.gesture_var.set(label)
            self.confidence_var.set(
                f"{confidence:.0%}"
            )
            self.status_var.set(status)

        # Handle events from the camera worker
        while True:
            try:
                event_type, message = self.event_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "status":
                self.status_var.set(message)

            elif event_type == "action":
                self.last_action_var.set(message)
                self.add_history(f"SENT: {message}")

            elif event_type == "log":
                self.add_history(message)

            elif event_type == "error":
                self.status_var.set("Error")
                messagebox.showerror(
                    "GesturePilot AI",
                    message
                )

            elif event_type == "stopped":
                self.status_var.set(message)

                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")

                self.preview.configure(
                    image="",
                    text="Camera is stopped\n\nClick Start Camera to begin"
                )
                self.preview.image = None

                self.gesture_var.set("Waiting")
                self.confidence_var.set("0%")

        # Recover buttons if the worker exits unexpectedly
        if self.worker is not None and not self.worker.is_alive():
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

        self.root.after(50, self.refresh_dashboard)

    # --------------------------------------------------
    # CLEAN SHUTDOWN
    # --------------------------------------------------

    def close_app(self):

        self.closing = True
        self.stop_event.set()
        self.commands_enabled.clear()

        self.wait_for_worker()

    def wait_for_worker(self):

        if self.worker is not None and self.worker.is_alive():
            self.root.after(100, self.wait_for_worker)
        else:
            self.root.destroy()


# ==================================================
# START THE APPLICATION
# ==================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GesturePilotDashboard(root)
    root.mainloop()
