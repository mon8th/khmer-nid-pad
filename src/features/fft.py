
import cv2
import numpy as np

from src.features.feature_input import FeatureInput


class FFT(FeatureInput):
    name = "fft"
    BINS = 64

    def compute(self):
        gray = self.gray.astype(np.float32)
        h, w = gray.shape
        gray -= gray.mean()
        window = np.outer(
            np.hanning(h),
            np.hanning(w)
        )
        gray *= window

        spectrum = np.fft.fftshift(
            np.fft.fft2(gray)
        )

        magnitude = np.log1p(np.abs(spectrum))

        # Distance from frequency center
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]

        radius = np.sqrt(
            (yy - cy) ** 2 +
            (xx - cx) ** 2
        )

        max_radius = min(cy, cx)

        rings = np.minimum(
            (radius / max_radius * self.BINS).astype(int),
            self.BINS - 1
        )

        radial = np.bincount(
            rings.ravel(),
            weights=magnitude.ravel(),
            minlength=self.BINS
        )

        counts = np.bincount(
            rings.ravel(),
            minlength=self.BINS
        )

        radial /= np.maximum(counts, 1)
        radial /= radial.sum() + 1e-9

        # Detect strong frequency peaks
        smooth = cv2.GaussianBlur(
            magnitude, (0, 0), 5
        )

        residual = magnitude - smooth
        values = residual[
            radius > 0.08 * max_radius
        ]

        std = values.std()

        if std < 1e-9:
            peak_stats = np.zeros(5)
        else:
            z = (values - values.mean()) / std

            top_count = max(int(0.001 * z.size), 8)
            top = np.sort(z)[-top_count:]

            kurtosis = np.mean(z ** 4) - 3

            peak_stats = np.array([
                z.max(),
                top.mean(),
                np.percentile(z, 99.9),
                (z > 4).mean() * 1000,
                kurtosis
            ])

        return np.concatenate([
            radial,
            peak_stats
        ]).astype(np.float32)
