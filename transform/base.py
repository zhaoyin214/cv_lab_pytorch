from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List

from common.define import ImageSize, Point, Sample
from utils.image.size import convert_size
from .random import RandomAngle, RandomChoice, RandomOrig, RandomScale

#-- interfaces --#
class ICrop(metaclass=ABCMeta):
    """the interface of the cropping operation"""
    def __init__(self, output_size: ImageSize) -> None:
        self._w, self._h = convert_size(output_size)

    @abstractmethod
    def _oper(self, sample: Any, orig: Point) -> Any:
        pass


class IScale(metaclass=ABCMeta):
    """the interface of the scaling operation"""
    @abstractmethod
    def _oper(self, sample: Any, scale: float) -> Any:
        pass


class IRotate(metaclass=ABCMeta):
    """the interface of the rotating operation"""
    @abstractmethod
    def _oper(self, sample: Any, angle: float) -> Any:
        pass


#-- basic transformers --#
class BaseTransformer(object):

    def __init__(self) -> None:
        # operators
        self._ops: List[Callable] = []

    def add_op(self, op: Callable) -> None:
        self._ops.append(op)

    def __call__(self, sample: Sample) -> Sample:
        for op in self._ops:
            sample = op(sample)

        return sample


class RandomTransformer(BaseTransformer):

    def __init__(self, prob: float) -> None:
        super(RandomTransformer, self).__init__()
        self._random_choice = RandomChoice(prob)

    def __call__(self, sample: Sample) -> Sample:

        if self._random_choice():
            sample = super(RandomTransformer, self).__call__(sample)

        return sample


class RandomCropTransformer(BaseTransformer):
    """
    op: ICrop
    """
    def __init__(self, output_size: ImageSize) -> None:
        super(RandomCropTransformer, self).__init__()
        self._random_orig = RandomOrig()
        self._w, self._h = convert_size(output_size)

    def __call__(self, sample: Sample) -> Sample:
        h, w = sample["image"].shape[: 2]
        x_range = max(1, w - self._w)
        y_range = max(1, h - self._h)
        x, y = self._random_orig(x_range, y_range)
        for op in self._ops:
            sample = op(sample, orig=(x, y))

        return sample


class RandomRotateTransformer(BaseTransformer):
    """
    op: IRotate
    """
    def __init__(self, max_angle: float) -> None:
        super(RandomRotateTransformer, self).__init__()
        self._random_angle = RandomAngle(max_angle)

    def __call__(self, sample: Sample) -> Sample:

        angle = self._random_angle()
        for op in self._ops:
            sample = op(sample, angle=angle)

        return sample


class RandomScaleTransformer(BaseTransformer):
    """
    op: IScale
    """
    def __init__(self, min_scale: float, max_scale: float) -> None:
        super(RandomScaleTransformer, self).__init__()
        self._random_scale = RandomScale(min_scale, max_scale)

    def __call__(self, sample: Sample) -> Sample:

        scale = self._random_scale()
        for op in self._ops:
            sample = op(sample, scale=scale)

        return sample

