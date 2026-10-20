from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "data-processed"
FOLDERS = ["raw", "screen", "printed"]
OUT_W, OUT_H = 640, 400
BRIGHTNESS_VARIANTS = {
    "normal": 1.0,
    "low": 0.7,
    "high": 1.3,
}


def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    for folder in FOLDERS:
        input_folder = DATA_DIR / folder
        output_folder = OUTPUT_DIR / folder
        if not input_folder.exists():
            continue
        output_folder.mkdir(parents=True, exist_ok=True)
        image_number = 1

        for image_path in sorted(input_folder.rglob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            result = cv2.resize(image, (OUT_W, OUT_H))

            written = True
            for variant, factor in BRIGHTNESS_VARIANTS.items():
                adjusted = adjust_brightness(result, factor)
                suffix = "" if variant == "normal" else f"_{variant}"
                filename = f"{folder}_processed_{image_number:02d}{suffix}.jpg"
                output_path = output_folder / filename
                written = cv2.imwrite(str(output_path), adjusted) and written

            if written:
                image_number += 1

    print("Finished processing images.")


if __name__ == "__main__":
    main()
