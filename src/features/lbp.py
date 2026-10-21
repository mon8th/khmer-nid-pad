import numpy as np
from skimage.feature import local_binary_pattern
from src.features.feature_input import FeatureInput


class LBP(FeatureInput):
    name = "lbp_simple"

    RADIUS, POINTS = 1, 8
    BINS = POINTS + 2

    GRID_ROWS = 4
    GRID_COLS = 5

    def compute(self):
        self.codes = local_binary_pattern(
            self.gray,
            self.POINTS,
            self.RADIUS,
            method="uniform"
        )

        h, w = self.codes.shape
        out = []

        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):

                patch = self.codes[
                    row * h // self.GRID_ROWS:
                    (row + 1) * h // self.GRID_ROWS,

                    col * w // self.GRID_COLS:
                    (col + 1) * w // self.GRID_COLS
                ]

                hist, _ = np.histogram(
                    patch,
                    bins=self.BINS,
                    range=(0, self.BINS)
                )
            
                hist = hist.astype(np.float32)
                hist /= hist.sum() + 1e-9

                out.append(hist)

        return np.concatenate(out).astype(np.float32)