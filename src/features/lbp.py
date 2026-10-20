import numpy as np
from skimage.feature import local_binary_pattern
from src.features.feature_input import FeatureInput

class LBP(FeatureInput):
    name = "lbp"
    RADIUS, POINTS = 1, 8
    BINS = POINTS + 2
    ROWS, COLS = 4, 5

    def compute(self):
        self.codes = local_binary_pattern(
            self.gray, self.POINTS, self.RADIUS, "uniform"
        )
        h, w = self.codes.shape
        ph, pw = h // self.ROWS, w // self.COLS
        return np.concatenate(
            [
                np.histogram(
                    self.codes[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw],
                    bins=self.BINS,
                    range=(0, self.BINS),
                    density=True,
                )[0]
                for r in range(self.ROWS)
                for c in range(self.COLS)
            ]
        )
