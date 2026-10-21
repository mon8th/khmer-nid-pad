
from pathlib import Path
import csv
import json

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features.hsv import HSVMoment
from src.features.lbp import LBP
from src.features.fft import FFT


# ========================================
# 1. SETTINGS
# ========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data-processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

METADATA_FILE = DATA_DIR / "card_metadata.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIZE = (640, 400)

LABELS = ["real", "screen", "printed"]

FOLDERS = {
    "raw": "real",
    "screen": "screen",
    "printed": "printed",
}

EXTENSIONS = {".jpg", ".jpeg", ".png"}

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ========================================
# 2. LOAD CARD METADATA
# ========================================

def load_metadata():
    """
    Read physical card IDs from CSV.

    Expected CSV columns:
        filename,card_id
    """

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}\n"
            "Create card_metadata.csv before training."
        )

    metadata = {}

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not {"filename", "card_id"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "CSV must contain filename and card_id columns."
            )

        for row in reader:
            filename = row["filename"].strip().replace("\\", "/")
            card_id = row["card_id"].strip()

            if not filename or not card_id:
                raise ValueError("Empty filename or card_id in CSV.")

            if filename in metadata:
                raise ValueError(
                    f"Duplicate metadata entry: {filename}"
                )

            metadata[filename] = card_id

    return metadata


# ========================================
# 3. EXTRACT FEATURES
# ========================================

def extract_features(image):
    """
    Return HSV, LBP and FFT feature vectors.
    """

    image = cv2.resize(image, SIZE)

    hsv = np.asarray(
        HSVMoment(image).values,
        dtype=np.float32
    ).ravel()

    lbp = np.asarray(
        LBP(image).values,
        dtype=np.float32
    ).ravel()

    fft = np.asarray(
        FFT(image).values,
        dtype=np.float32
    ).ravel()

    if len(hsv) != 244:
        raise ValueError(
            f"Expected 244 HSV features, got {len(hsv)}"
        )

    if len(lbp) != 200:
        raise ValueError(
            f"Expected 200 LBP features, got {len(lbp)}"
        )

    if len(fft) != 69:
        raise ValueError(
            f"Expected 69 FFT features, got {len(fft)}"
        )

    features = {
        "hsv": hsv,
        "lbp": lbp,
        "fft": fft,
    }

    for name, values in features.items():
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite {name} features detected."
            )

    return features


# ========================================
# 4. LOAD DATASET
# ========================================

def load_dataset():
    """
    Load all images and extract their features.

    Also record:
        y       = class label
        groups  = physical card ID
    """

    metadata = load_metadata()

    hsv_data = []
    lbp_data = []
    fft_data = []

    labels = []
    groups = []

    image_count = 0

    for folder, label in FOLDERS.items():

        folder_path = DATA_DIR / folder

        if not folder_path.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder_path}"
            )

        for image_path in sorted(folder_path.rglob("*")):

            if image_path.suffix.lower() not in EXTENSIONS:
                continue

            relative_path = image_path.relative_to(
                DATA_DIR
            ).as_posix()

            if relative_path not in metadata:
                raise ValueError(
                    f"Missing card_id for: {relative_path}\n"
                    "Add this image to card_metadata.csv."
                )

            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError(
                    f"Cannot read image: {image_path}"
                )

            features = extract_features(image)

            hsv_data.append(features["hsv"])
            lbp_data.append(features["lbp"])
            fft_data.append(features["fft"])

            labels.append(label)
            groups.append(metadata[relative_path])

            image_count += 1

    if image_count == 0:
        raise RuntimeError("No images found.")

    hsv_data = np.asarray(hsv_data, dtype=np.float32)
    lbp_data = np.asarray(lbp_data, dtype=np.float32)
    fft_data = np.asarray(fft_data, dtype=np.float32)

    y = np.asarray(labels)
    card_ids = np.asarray(groups)

    print("\n===== DATASET =====")
    print("Total images:", image_count)
    print("Unique physical cards:", len(np.unique(card_ids)))

    for label in LABELS:
        print(f"{label.upper()}: {np.sum(y == label)}")

    return hsv_data, lbp_data, fft_data, y, card_ids


# ========================================
# 5. TRAIN AND EVALUATE SVM
# ========================================

def train_model(
    name,
    X,
    y,
    train_idx,
    test_idx
):
    """
    Train an SVM and evaluate on held-out cards.
    """

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    if len(np.unique(y_train)) < 3:
        raise ValueError(
            "Training set must contain all three classes."
        )

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        )
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    print(f"\n===== {name.upper()} =====")
    print("Features:", X.shape[1])
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        zero_division=0
    )

    print(report)

    # Save classification report
    report_path = RESULTS_DIR / f"{name}_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{name.upper()} - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"{name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # Save trained model
    model_path = MODEL_DIR / f"{name}_svm.joblib"
    joblib.dump(model, model_path)

    print("Saved model:", model_path)

    return {
        "model": name,
        "features": int(X.shape[1]),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


# ========================================
# 6. MAIN PROGRAM
# ========================================

def main():

    hsv, lbp, fft, y, card_ids = load_dataset()

    # Feature combinations
    experiments = {
        "hsv": hsv,
        "lbp": lbp,
        "hsv_lbp": np.concatenate(
            [hsv, lbp], axis=1
        ),
        "hsv_lbp_fft": np.concatenate(
            [hsv, lbp, fft], axis=1
        ),
    }

    # Split by physical ID card
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(
            experiments["hsv_lbp_fft"],
            y,
            groups=card_ids
        )
    )

    # Check group separation
    train_cards = set(card_ids[train_idx])
    test_cards = set(card_ids[test_idx])

    assert train_cards.isdisjoint(test_cards)

    print("\n===== DATA SPLIT =====")
    print("Training images:", len(train_idx))
    print("Testing images:", len(test_idx))
    print("Training cards:", len(train_cards))
    print("Testing cards:", len(test_cards))

    for split_name, indices in [
        ("Training", train_idx),
        ("Testing", test_idx),
    ]:
        print(f"\n{split_name} classes:")

        for label in LABELS:
            print(
                f"  {label}: {np.sum(y[indices] == label)}"
            )

    # Train every model using the same split
    results = []

    for name, X in experiments.items():

        result = train_model(
            name,
            X,
            y,
            train_idx,
            test_idx
        )

        results.append(result)

    # Save summary
    summary_path = RESULTS_DIR / "svm_results.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Compare models
    names = [result["model"] for result in results]
    scores = [result["macro_f1"] for result in results]

    plt.figure(figsize=(10, 5))
    plt.bar(names, scores)
    plt.ylabel("Macro F1")
    plt.xlabel("Feature Configuration")
    plt.title("SVM Feature Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "svm_model_comparison.png",
        dpi=300
    )

    plt.close()

    print("\n===== FINISHED =====")

    for result in results:
        print(
            f"{result['model']:15} "
            f"Features: {result['features']:4} "
            f"Accuracy: {result['accuracy']:.4f} "
            f"Macro F1: {result['macro_f1']:.4f}"
        )

    print("\nResults saved to:", RESULTS_DIR)
    print("Models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
