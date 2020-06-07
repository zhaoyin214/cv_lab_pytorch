import numpy as np

from common.define import Point

class RandomOrig(object):

    def __init__(self) -> None:
        self._x = 0
        self._y = 0

    def update(self, x_range: int, y_range: int) -> None:
        self._x = np.random.randint(0, x_range)
        self._y = np.random.randint(0, y_range)

    def __call__(self) -> Point:
        return self._x, self._y
