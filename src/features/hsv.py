
import numpy as np
from src.features.feature_input import FeatureInput, Moment


class HSVMoment(FeatureInput):
    name = "hsv_moments"

    def compute(self):
        hsv = self.hsv
        out = []

        # 1. HSV Moments (12 features)
        for ch in range(3):
            pixels = hsv[..., ch].astype(np.float32)
            stats = Moment(pixels)

            out.extend([
                stats.mean,
                stats.std,
                stats.skewness,
                stats.kurtosis
            ])

        # 2. HSV Percentiles (9 features)
        for ch in range(3):
            pixels = hsv[..., ch]

            out.extend(
                np.percentile(pixels, [10, 50, 90])
            )

        # 3. Saturation Statistics (2 features)
        saturation = hsv[..., 1]

        out.extend([
            np.mean(saturation > 80),
            np.mean(saturation < 30)
        ])

        # 4. Brightness Statistics (2 features)
        brightness = hsv[..., 2]

        out.extend([
            np.mean(brightness < 20),
            np.mean(brightness > 235)
        ])

        # 5. HSV Entropy (3 features)
        for ch in range(3):
            upper = 180 if ch == 0 else 256

            hist, _ = np.histogram(
                hsv[..., ch],
                bins=32,
                range=(0, upper)
            )

            p = hist.astype(np.float64)
            p /= p.sum() + 1e-9
            p = p[p > 0]

            entropy = -np.sum(p * np.log2(p))
            out.append(entropy)

        # 6. HSV Histograms (96 features)
        for ch in range(3):
            upper = 180 if ch == 0 else 256

            hist, _ = np.histogram(
                hsv[..., ch],
                bins=32,
                range=(0, upper)
            )

            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-9

            out.extend(hist)

        # 7. Spatial HSV Features (120 features)
        height, width = hsv.shape[:2]

        for row in range(4):
            for col in range(5):
                patch = hsv[
                    row * height // 4:(row + 1) * height // 4,
                    col * width // 5:(col + 1) * width // 5
                ]

                for ch in range(3):
                    pixels = patch[..., ch].astype(np.float32)

                    out.extend([
                        pixels.mean(),
                        pixels.std()
                    ])

        return np.array(out, dtype=np.float32)