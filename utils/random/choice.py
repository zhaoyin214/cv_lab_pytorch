import numpy as np

class RandomChoice(object):
    """"uniform-randomly choice with a given probability"""
    def __init__(self, prob: float=0.5) -> None:
        self._prob = prob
        self._choice = True

    def update(self) -> None:
        self._choice = float(np.random.rand(1)) < self._prob

    def __call__(self) -> bool:
        return self._choice
