
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


# File locations
DATA_PATH = Path(__file__).with_name("gesture_data.csv")
MODEL_PATH = Path(__file__).with_name("gesture_model.joblib")

FEATURES = [f"f{i}" for i in range(63)]

EXPECTED_GESTURES = [
    "open_palm",
    "fist",
    "peace",
    "thumbs_up"
]


# 1. Check whether the dataset exists
if not DATA_PATH.exists():
    raise SystemExit(
        "Dataset not found. Run collect_data.py first."
    )

# 2. Load the dataset
data = pd.read_csv(DATA_PATH)

required_columns = ["label"] + FEATURES

if not all(col in data.columns for col in required_columns):
    raise SystemExit(
        "Dataset columns are incorrect. "
        "Please check gesture_data.csv."
    )

data = data.dropna(subset=required_columns)
data["label"] = data["label"].astype(str).str.strip()

print("\nSamples for each gesture:")
print(data["label"].value_counts())

# 3. Ensure every gesture has enough samples
counts = data["label"].value_counts()

for gesture in EXPECTED_GESTURES:
    if counts.get(gesture, 0) < 8:
        raise SystemExit(
            f"Not enough samples for {gesture}. "
            "Collect at least 8 samples per gesture; "
            "30 or more is recommended."
        )

# 4. Separate features and labels
X = data[FEATURES]
y = data["label"]

# 5. Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 6. Create and train the model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining the model...")
model.fit(X_train, y_train)

# 7. Evaluate on unseen test rows
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=model.classes_,
        zero_division=0
    )
)

print("Confusion matrix:")
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print(
    pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_
    )
)

# 8. Save the trained model and its labels
joblib.dump(
    {
        "model": model,
        "labels": model.classes_.tolist(),
        "features": FEATURES
    },
    MODEL_PATH
)

print(f"\nModel saved successfully: {MODEL_PATH.name}")
print("Next: connect the trained model to the webcam.")
