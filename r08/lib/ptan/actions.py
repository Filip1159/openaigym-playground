import numpy as np


class ArgmaxActionSelector:
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)
