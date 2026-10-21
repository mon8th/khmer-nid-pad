
import csv
import re
from pathlib import Path

csv_path = Path("data-processed/card_metadata.csv")

with open(csv_path, newline="") as file:
    rows = list(csv.DictReader(file))

for row in rows:
    filename = Path(row["filename"])
    folder = filename.parent.name

    # Extract the original image number
    match = re.search(
        r"_processed_(\d+)(?:_high|_low)?$",
        filename.stem
    )

    if not match:
        raise ValueError(f"Unexpected filename: {filename}")

    number = int(match.group(1))

    # Temporary source-group ID
    row["card_id"] = f"{folder}_{number:03d}"

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "card_id"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Grouped image variants successfully.")
print("Review card IDs before training.")
