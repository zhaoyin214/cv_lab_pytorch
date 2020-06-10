from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List

from common.define import ImageSize, Point, Sample
from utils.image import convert_size
from utils.box import BoxWrapper
from .random import RandomAngle, RandomChoice, RandomOrig, RandomScale


class IProc(metaclass=ABCMeta):
    """the template of processors"""
    @abstractmethod
    def _oper(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def __call__(self, sample: Sample, *args, **kwargs) -> Sample:
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
    op: Crop
    """
    def __init__(self, output_size: ImageSize) -> None:
        super(RandomCropTransformer, self).__init__()
        self._random_orig = RandomOrig()
        self._output_w, self._output_h = convert_size(output_size)
        self._box_wrapper = BoxWrapper()

    def __call__(self, sample: Sample) -> Sample:
        h, w = sample["image"].shape[: 2]
        x_range = max(1, w - self._output_w)
        y_range = max(1, h - self._output_h)
        x, y = self._random_orig(x_range, y_range)
        for op in self._ops:
            self._box_wrapper.set_box_ltwh(
                x, y, self._output_w, self._output_h
            )

            sample = op(sample, roi=self._box_wrapper.box)

        return sample


class RandomRotateTransformer(BaseTransformer):
    """
    op: Rotate
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
    op: Scale
    """
    def __init__(self, min_scale: float, max_scale: float) -> None:
        super(RandomScaleTransformer, self).__init__()
        self._random_scale = RandomScale(min_scale, max_scale)

    def __call__(self, sample: Sample) -> Sample:

        scale = self._random_scale()
        for op in self._ops:
            sample = op(sample, scale=scale)

        return sample

