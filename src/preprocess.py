from pathlib import Path
import argparse
import csv

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "data-processed"
CARDS_CSV = DATA_DIR / "cards.csv"
METADATA_CSV = OUTPUT_DIR / "card_metadata.csv"
SIZE = (640, 400)
LABEL_FOLDERS = {"real": "raw", "screen": "screen", "printed": "printed"}
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def read_cards():
    if not CARDS_CSV.exists():
        raise FileNotFoundError(f"Missing {CARDS_CSV}")

    with open(CARDS_CSV, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    seen = set()
    for row in rows:
        path, label, card_id = row["path"].strip(), row["label"].strip(), row["card_id"].strip()
        if label not in LABEL_FOLDERS:
            raise ValueError(f"Unknown label '{label}' for {path} (use real/screen/printed)")
        if not card_id:
            raise ValueError(f"Missing card_id for {path}")
        if path in seen:
            raise ValueError(f"Duplicate row: {path}")
        seen.add(path)
        row.update(path=path, label=label, card_id=card_id)
    return rows


def add_unlisted(rows):
    folder_labels = {folder: label for label, folder in LABEL_FOLDERS.items()}
    listed = {row["path"] for row in rows}
    added = []

    for folder, label in folder_labels.items():
        for image_path in sorted((DATA_DIR / folder).rglob("*")):
            relative = image_path.relative_to(DATA_DIR).as_posix()
            if image_path.suffix.lower() in EXTENSIONS and relative not in listed:
                card_id = "auto_" + relative.rsplit(".", 1)[0].replace("/", "_").replace(" ", "_")
                added.append({"path": relative, "label": label, "card_id": card_id})

    if added:
        rows = rows + added
        with open(CARDS_CSV, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "label", "card_id"])
            writer.writeheader()
            writer.writerows({k: row[k] for k in ("path", "label", "card_id")} for row in rows)
        print(f"Added {len(added)} new photos to {CARDS_CSV.name}, each with its own auto_ card_id.")
        print("If several photos show the same physical card, give them the same card_id there.")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, nargs=2, default=SIZE, metavar=("W", "H"))
    args = parser.parse_args()
    size = tuple(args.size)

    rows = add_unlisted(read_cards())

    for folder in LABEL_FOLDERS.values():
        out = OUTPUT_DIR / folder
        out.mkdir(parents=True, exist_ok=True)
        for old in out.iterdir():
            if old.suffix.lower() in EXTENSIONS:
                old.unlink()

    metadata = []
    for row in rows:
        image = cv2.imread(str(DATA_DIR / row["path"]))
        if image is None:
            raise ValueError(f"Cannot read image: {row['path']}")

        source = Path(row["path"])
        parent = "" if source.parent.name == LABEL_FOLDERS[row["label"]] else source.parent.name + "_"
        relative = f"{LABEL_FOLDERS[row['label']]}/{row['card_id']}_{parent}{source.stem}.png"

        if not cv2.imwrite(str(OUTPUT_DIR / relative), cv2.resize(image, size, interpolation=cv2.INTER_AREA)):
            raise IOError(f"Cannot write {relative}")
        metadata.append({"filename": relative, "card_id": row["card_id"]})

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "card_id"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Wrote {len(metadata)} images at {size[0]}x{size[1]} and {METADATA_CSV.name}")
    for label, folder in LABEL_FOLDERS.items():
        print(f"  {label:8s} {sum(m['filename'].startswith(folder + '/') for m in metadata)}")
    if size != SIZE:
        print(f"Note: set SIZE = {size} in src/preprocess.py so svm.py and predict.py use it too.")


if __name__ == "__main__":
    main()
