import numpy as np


class ArgmaxActionSelector:
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.argmax(scores, axis=1)
