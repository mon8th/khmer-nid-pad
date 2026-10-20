import cv2
import numpy as np


class FeatureInput:
    """
    Base for every feature: pass a BGR image (from cv2.imread), 
    get .values as a flat float32 vector.
    Subclasses only implement compute()
    """
    name = "feature"

    def __init__(self, x):
        if x is None:
            raise ValueError("got None — did cv2.imread fail to read the file?")
        self.image = np.asarray(x)
        self.values = np.asarray(self.compute()).ravel().astype(np.float32)

    @property
    def gray(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    @property
    def hsv(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def __len__(self):
        return self.values.size

    def __repr__(self):
        return f"{type(self).__name__}(dims={len(self)})"

    def compute(self):
        raise NotImplementedError(f"{type(self).__name__} must implement compute()")

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.bar(range(len(self)), self.values, width=1)
        ax.set_title(f"{self.name} ({len(self)} dims)")
        return ax


class Moment(FeatureInput):
    """
    mean, std, skewness (lopsidedness),
    kurtosis (tail weight) of any array
    """
    name = "moment"

    def compute(self):
        x = self.image.ravel().astype(np.float64)
        c = x - x.mean()
        var = (c * c).mean()
        std = (var + 1e-12) ** 0.5
        return [x.mean(), std, (c**3).mean() / std**3, (c**4).mean() / (var * var + 1e-12) - 3.0]

    mean = property(lambda s: s.values[0])
    std = property(lambda s: s.values[1])
    skewness = property(lambda s: s.values[2])
    kurtosis = property(lambda s: s.values[3])


def combine(img, *feature_classes):
    return np.concatenate([F(img).values for F in feature_classes])