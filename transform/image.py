from abc import ABCMeta, abstractmethod
from typing import Dict, List
import cv2
import numpy as np
import torch

from common.define import Image, ImageSize, Sample, Point
from utils.image.size import convert_resize, convert_size
from utils.visual import show_image
from .base import \
    ICrop, \
    RandomCropTransformer, RandomRotateTransformer, RandomScaleTransformer, \
    RandomTransformer
from .random import RandomChoice, RandomOrig

__all__ = [
    "RandomBlur", "RandomHorizontalFlip", "RandomCrop", "RandomRotate",
    "RandomScale", "Resize", "ToTensor"
]


class IProcImage(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, image: Image) -> Image:
        pass


class ProcImage(object):

    def __init__(self, op: IProcImage, **kwarg):
        self._op = op
        self._kwarg = kwarg

    def __call__(self, sample: Sample, **kwarg) -> Sample:

        image = sample["image"]
        image = self._op(image, **self._kwarg, **kwarg)
        sample["image"] = image

        return sample


class Blur(ProcImage):
    """blur an image with the given sigma

    arguments:
        ksize: kernel size
    """
    def __init__(self, ksize: int=3) -> None:
        assert ksize % 2 == 1
        super(Blur, self).__init__(cv2.blur, ksize=(ksize, ksize))


class HorizontalFlip(ProcImage):
    """flip an input image and landmarks horizontally"""
    def __init__(self) -> None:
        super(HorizontalFlip, self).__init__(cv2.flip, flipCode=1)


class Crop(ICrop, ProcImage):
    """make a crop from the source image"""
    def __init__(self, output_size: ImageSize) -> None:
        self._w, self._h = convert_size(output_size)
        super(Crop, self).__init__(self._oper)

    def _oper(self, image: Image, orig: Point) -> Image:
        left, top = orig
        image = image[
            top : top + self._h, left : left + self._w, :
        ]

        return image


class Rotate(ProcImage):
    """rotate an image around it's center by a given angle"""
    def __init__(self) -> None:
        super(Rotate, self).__init__(self._oper)

    def _oper(self, image: Image, angle: float) -> Image:
        h, w = image.shape[:2]
        rot_mat = cv2.getRotationMatrix2D(
            center=(w * 0.5, h * 0.5), angle=angle, scale=1
        )
        image = cv2.warpAffine(
            src=image, M=rot_mat, dsize=(w, h), flags=cv2.INTER_LANCZOS4
        )

        return image


class Scale(ProcImage):
    """perform scale"""
    def __init__(self) -> None:
        super(Scale, self).__init__(self._oper)

    def _oper(self, image: Image, scale: float) -> Image:
        h, w = image.shape[:2]
        scale_mat = cv2.getRotationMatrix2D(
            center=(w * 0.5, h * 0.5), angle=0, scale=scale
        )
        image = cv2.warpAffine(
            src=image, M=scale_mat, dsize=(w, h), flags=cv2.INTER_LANCZOS4
        )

        return image


class RandomBlur(RandomTransformer):
    """blur an image with a given sigma and probability"""
    def __init__(self, prob: float=0.3, ksize: int=3) -> None:
        super(RandomBlur, self).__init__(prob)
        blur = Blur(ksize=ksize)
        self.add_op(blur)


class RandomHorizontalFlip(RandomTransformer):
    """flip an input image and landmarks horizontally with a given probability
    """
    def __init__(self, prob: float=0.5) -> None:
        super(RandomHorizontalFlip, self).__init__(prob)
        flip = HorizontalFlip()
        self.add_op(flip)


class RandomCrop_(RandomCropTransformer):
    """make a random crop from the source image"""
    def __init__(self, output_size: ImageSize) -> None:
        super(RandomCrop_, self).__init__(output_size)
        crop = Crop(output_size)
        self.add_op(crop)


class RandomCrop(RandomTransformer):
    """make a random crop from the source image with a given probability"""
    def __init__(self, output_size: ImageSize, prob: float=0.5) -> None:
        super(RandomCrop, self).__init__(prob)
        crop = RandomCrop_(output_size)
        self.add_op(crop)


class RandomRotate_(RandomRotateTransformer):
    """rotate an image around it's center by a random angle"""
    def __init__(self, max_angle: float) -> None:
        super(RandomRotate_, self).__init__(max_angle)
        rotate = Rotate()
        self.add_op(rotate)


class RandomRotate(RandomTransformer):
    """rotate an image around it's center by a random angl with a given probability"""
    def __init__(self, max_angle: float=30, prob: float=0.5) -> None:
        super(RandomRotate, self).__init__(prob)
        rotate = RandomRotate_(max_angle)
        self.add_op(rotate)


class RandomScale_(RandomScaleTransformer):
    """perform uniform scale with a random magnitude"""
    def __init__(self, min_scale: float, max_scale: float) -> None:
        super(RandomScale_, self).__init__(min_scale, max_scale)
        scale = Scale()
        self.add_op(scale)


class RandomScale(RandomTransformer):
    """perform uniform scale with a random magnitude with a given probability"""
    def __init__(
        self, min_scale: float=0.5, max_scale: float=1.5, prob: float=0.3
    ) -> None:
        super(RandomScale, self).__init__(prob)
        scale = RandomScale_(min_scale, max_scale)
        self.add_op(scale)


class Resize(ProcImage):
    """resize an image"""
    def __init__(self, output_size: ImageSize) -> None:
        self._output_size = output_size
        super(Resize, self).__init__(self._resize)

    def _resize(self, image: Image) -> Image:

        h, w = image.shape[:2]
        output_size = convert_resize(
            input_size=(h, w), output_size=self._output_size
        )
        image = cv2.resize(image, output_size)

        return image


class Show(ProcImage):
    """show image using opencv"""
    def __init__(self) -> None:
        super(Show, self).__init__(self._show)

    def _show(self, image: Image) -> Image:
        show_image(image, win_name="")
        return image


class ToTensor(ProcImage):
    """convert a ndarray to a tensor with the range of [0, 1]"""
    def __init__(self, switch_rb: bool=False) -> None:
        self._switch_rb = switch_rb
        super(ToTensor, self).__init__(self._convert)

    def _convert(self, image: Image) -> Image:
        # numpy image: H x W x C
        # torch image: C X H X W
        if self._switch_rb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).type(torch.FloatTensor) / 255


if __name__ == "__main__":

    image = cv2.imread("./img/31.jpg")
    sample = {"image": image}

    # # blur
    # blur = Blur()
    # show_image(blur(sample)["image"])

    # # flip
    # flip = HorizontalFlip()
    # show_image(flip(sample)["image"])

    # # crop
    # crop = Crop(output_size=400)
    # show_image(crop(sample, orig=(200, 50))["image"])

    # # rotate
    # rot = Rotate()
    # show_image(rot(sample, angle=30)["image"])

    # # resize
    # resize = Resize((500, 300))
    # show_image(resize(sample)["image"])

    # # show
    # Show()(sample)

    # # tensor
    # tensor = ToTensor()
    # sample = tensor(sample)
    # print(type(sample["image"]))

    # random_blur = RandomBlur(prob=0.3, ksize=5)
    # show_image(random_blur(sample)["image"])

    # random_flip = RandomHorizontalFlip()
    # show_image(random_flip(sample)["image"])

    # random_crop_ = RandomCrop_((500, 300))
    # show_image(random_crop(sample)["image"])

    # random_crop = RandomCrop((500, 300))
    # show_image(random_crop(sample)["image"])

    # random_rot_ = RandomRotate_(30)
    # show_image(random_rot_(sample)["image"])

    random_rot = RandomRotate()
    show_image(random_rot(sample)["image"])

    # random_scale_ = RandomScale_(0.5, 1.5)
    # show_image(random_scale_(sample)["image"])

    # random_scale = RandomScale()
    # show_image(random_scale(sample)["image"])
