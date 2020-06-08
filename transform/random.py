import numpy as np

from common.define import Point

class RandomChoice(object):
    """"uniform-randomly choice with a given probability"""
    def __init__(self, prob: float) -> None:
        self._prob = prob

    def __call__(self) -> bool:
        return float(np.random.rand(1)) < self._prob


class RandomOrig(object):
    """"uniformly sample a point in the area [[0, x_range], [0, y_range]]"""

    def __call__(self, x_range: int, y_range: int) -> Point:
        x = np.random.randint(0, x_range)
        y = np.random.randint(0, y_range)

        return x, y


class RandomAngle(object):
    """"uniformly sample an angle in the range [- max_angle, max_angle]"""

    def __init__(self, max_angle: float) -> None:
        self._max_angle = max_angle

    def __call__(self) -> float:
        angle = \
            np.random.choice([-1, 1]) * \
            float(np.random.rand(1)) * \
            self._max_angle

        return angle


class RandomScale(object):
    """"uniformly sample an angle in the range [- max_angle, max_angle]"""

    def __init__(self, min_scale: float, max_scale: float) -> None:
        self._min_scale = min_scale
        self._max_scale = max_scale

    def __call__(self) -> float:
        scale = self._min_scale + \
            float(np.random.rand(1)) * (self._max_scale - self._min_scale)

        return scale