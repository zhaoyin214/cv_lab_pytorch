from abc import abstractmethod
from torchvision import transforms
from typing import Tuple
import cv2
import numpy as np
import torch

from common.define import Box, Image, ImageSize, Sample, TensorImage
from utils.image import crop as image_crop
from utils.image import convert_resize
from utils.visual import show_image
from .base import IProc, \
    RandomCropTransformer, RandomRotateTransformer, RandomScaleTransformer, \
    RandomTransformer
from .random import RandomChoice, RandomOrig

__all__ = [
    "RandomBlur", "RandomHorizontalFlip", "RandomCrop", "RandomRotate",
    "RandomScale", "Resize", "ToTensor", "Normalize"
]


class ProcImage(IProc):
    """the template of image processors"""
    @abstractmethod
    def _oper(self, image: Image, *args, **kwargs) -> Image:
        pass

    def __call__(self, sample: Sample, *args, **kwargs) -> Sample:

        image = sample["image"]
        image = self._oper(image, *args, **kwargs)
        sample["image"] = image

        return sample


class Blur(ProcImage):
    """blur an image with the given sigma

    arguments:
        ksize: kernel size
    """
    def __init__(self, ksize: int=3) -> None:
        assert ksize % 2 == 1
        self._ksize = (ksize, ksize)

    def _oper(self, image: Image) -> Image:
        return cv2.blur(image, ksize=self._ksize)


class HorizontalFlip(ProcImage):
    """flip an input image and landmarks horizontally"""
    def _oper(self, image: Image) -> Image:
        return cv2.flip(image, flipCode=1)


class Crop(ProcImage):
    """make a crop from the source image"""
    def _oper(self, image: Image, roi: Box) -> Image:
        image = image_crop(image, roi)

        return image


class Rotate(ProcImage):
    """rotate an image around it's center by a given angle"""
    def _oper(self, image: Image, angle: float) -> Image:
        h, w = image.shape[:2]
        rot_mat = cv2.getRotationMatrix2D(
            center=(w * 0.5, h * 0.5), angle=angle, scale=1
        )
        image = cv2.warpAffine(
            src=image, M=rot_mat, dsize=(w, h), flags=cv2.INTER_LANCZOS4
        )

        return image


class Resize(ProcImage):
    """resize an image"""
    def __init__(self, output_size: ImageSize) -> None:
        self._output_size = output_size

    def _oper(self, image: Image) -> Image:

        h, w = image.shape[:2]
        output_size = convert_resize(
            input_size=(h, w), output_size=self._output_size
        )
        image = cv2.resize(image, output_size)

        return image


class Scale(ProcImage):
    """perform scale"""
    def _oper(self, image: Image, scale: float) -> Image:
        h, w = image.shape[:2]
        scale_mat = cv2.getRotationMatrix2D(
            center=(w * 0.5, h * 0.5), angle=0, scale=scale
        )
        image = cv2.warpAffine(
            src=image, M=scale_mat, dsize=(w, h), flags=cv2.INTER_LANCZOS4
        )

        return image


class Show(object):
    """show image using opencv"""
    def __call__(self, sample: Sample) -> Sample:
        show_image(sample["image"], win_name="")
        return sample


class ToTensor(ProcImage):
    """convert a ndarray to a tensor with the range of [0, 1]"""
    def __init__(self, switch_rb: bool=True) -> None:
        self._switch_rb = switch_rb
        self._to_tensor = transforms.ToTensor()

    def _oper(self, image: Image) -> TensorImage:
        # numpy image: H x W x C
        # torch image: C X H X W
        if self._switch_rb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = self._to_tensor(image)

        return image


class Normalize(ProcImage):
    """normalization with a give mean and std (R, G, B)"""
    def __init__(
        self,
        mean: Tuple=(0.485, 0.456, 0.406),
        std: Tuple=(0.229, 0.224, 0.225)
    ) -> None:
        self._normalizer = transforms.Normalize(mean, std)

    def _oper(self, image: TensorImage) -> TensorImage:
        image = self._normalizer(image)

        return image


class RandomBlur(RandomTransformer):
    """blur an image with a given sigma and probability"""
    def __init__(self, prob: float=0.3, ksize: int=3) -> None:
        super(RandomBlur, self).__init__(prob)
        blur = Blur(ksize=ksize)
        self.add_op(blur)


class RandomHorizontalFlip(RandomTransformer):
    """flip an input image horizontally with a given probability
    """
    def __init__(self, prob: float=0.5) -> None:
        super(RandomHorizontalFlip, self).__init__(prob)
        flip = HorizontalFlip()
        self.add_op(flip)


class RandomCrop_(RandomCropTransformer):
    """make a random crop from the source image"""
    def __init__(self, output_size: ImageSize) -> None:
        super(RandomCrop_, self).__init__(output_size)
        crop = Crop()
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
    """rotate an image around it's center by a random angle
       with a given probability
    """
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


if __name__ == "__main__":

    image = cv2.imread("./img/31.jpg")
    sample = {"image": image}

    # blur
    blur = Blur()
    show_image(blur(sample)["image"])

    # flip
    flip = HorizontalFlip()
    show_image(flip(sample)["image"])

    # crop
    crop = Crop()
    show_image(crop(sample, roi=(200, 50, 499, 349))["image"])
    print(sample["image"].shape)

    # rotate
    rot = Rotate()
    show_image(rot(sample, angle=30)["image"])

    # resize
    resize = Resize((500, 300))
    show_image(resize(sample)["image"])

    # show
    Show()(sample)

    # to tensor
    tensor = ToTensor()
    sample = tensor(sample)
    print(type(sample["image"]))

    # normalize
    normalizer = Normalize()
    sample = normalizer(sample)
    print(sample["image"])


    # random
    image = cv2.imread("./img/31.jpg")
    sample = {"image": image}

    random_blur = RandomBlur(prob=0.3, ksize=5)
    show_image(random_blur(sample)["image"])

    random_flip = RandomHorizontalFlip()
    show_image(random_flip(sample)["image"])

    random_crop_ = RandomCrop_((500, 300))
    show_image(random_crop_(sample)["image"])
    print(sample["image"].shape)

    random_crop = RandomCrop((500, 300))
    show_image(random_crop(sample)["image"])

    random_rot_ = RandomRotate_(30)
    show_image(random_rot_(sample)["image"])

    random_rot = RandomRotate()
    show_image(random_rot(sample)["image"])

    random_scale_ = RandomScale_(0.5, 1.5)
    show_image(random_scale_(sample)["image"])

    random_scale = RandomScale()
    show_image(random_scale(sample)["image"])
