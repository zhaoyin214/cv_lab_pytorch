import numpy as np

class RandomAngle(object):

    def __init__(self, max_angle: float) -> None:
        self._max_angle = max_angle

    def update(self) -> None:
        self._angle = \
            np.random.choice([-1, 1]) * \
            float(np.random.rand(1)) * \
            self._max_angle

    def __call__(self) -> float:
        return self._angle