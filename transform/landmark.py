import cv2
import numpy as np
import torch
from typing import Dict

from common.define import Box, ImageSize, Landmarks, Sample
from utils.landmark import LandmarksWrapper
from utils.visual import show_landmarks
from transform import image
from .base import IProc, \
    RandomCropTransformer, RandomRotateTransformer, RandomScaleTransformer, \
    RandomTransformer

__all__ = [
    "RandomBlur", "RandomHorizontalFlip", "RandomCrop", "RandomRotate",
    "RandomScale", "Resize", "ToTensor", "Normalize"
]


class ProcLandmarks(IProc):
    """the template of landmark processors, it's a decorator of an image processor

    note: a landmark processor should be called before an image processor,
          because `image_size` is the size of the input image before processing
    """
    def __init__(self, component: IProc) -> None:
        self._comp = component

    def _oper(
        self,
        landmarks_wrapper: LandmarksWrapper,
        image_size: ImageSize,
        *args,
        **kwargs
    ) -> LandmarksWrapper:

        return landmarks_wrapper

    def __call__(self, sample: Sample, *args, **kwargs) -> Sample:

        h, w = sample["image"].shape[: 2]
        image_size: ImageSize = (w, h)
        landmarks_wrapper = LandmarksWrapper(sample["landmarks"])
        landmarks_wrapper = self._oper(landmarks_wrapper, image_size, *args, **kwargs)
        sample["landmarks"] = landmarks_wrapper.landmarks
        self._comp(sample, *args, **kwargs)

        return sample


class Blur(ProcLandmarks):
    """blur an image"""
    def __init__(self) -> None:
        image_blur = image.Blur()
        super(Blur, self).__init__(image_blur)


class HorizontalFlip(ProcLandmarks):
    """flip an image and landmarks horizontally"""
    def __init__(self) -> None:
        image_flip = image.HorizontalFlip()
        super(HorizontalFlip, self).__init__(image_flip)

    def _oper(
        self, landmarks_wrapper: LandmarksWrapper, image_size: ImageSize
    ) -> LandmarksWrapper:
        landmarks_wrapper.horizontal_flip()
        return landmarks_wrapper


class Crop(ProcLandmarks):
    """make a crop from an image and
       perform the corresponding transformation on landmarks
    """
    def __init__(self) -> None:
        image_crop = image.Crop()
        super(Crop, self).__init__(image_crop)
    def _oper(
        self,
        landmarks_wrapper: LandmarksWrapper,
        image_size: ImageSize,
        roi: Box
    ) -> LandmarksWrapper:
        w, h = image_size
        left, top, right, bottom = roi
        output_w = right - left + 1
        output_h = bottom - top + 1

        landmarks_wrapper.horizontal_shift(- left / w)
        landmarks_wrapper.vertical_shift(- top / h)
        landmarks_wrapper.horizontal_scale(w / output_w)
        landmarks_wrapper.vertical_scale(h / output_h)

        return landmarks_wrapper


class Rotate(ProcLandmarks):
    """rotate an image around its center by a give angle
       perform the same transformation on landmarks
    """
    def __init__(self) -> None:
        image_rotate = image.Rotate()
        super(Rotate, self).__init__(image_rotate)
    def _oper(
        self,
        landmarks_wrapper: LandmarksWrapper,
        image_size: ImageSize,
        angle: float
    ) -> LandmarksWrapper:
        landmarks_wrapper.rotate(angle)

        return landmarks_wrapper


class Resize(ProcLandmarks):
    """resize an image and corresponding landmarks"""
    def __init__(self, output_size: ImageSize) -> None:
        image_resize = image.Resize(output_size)
        super(Resize, self).__init__(image_resize)


class Scale(ProcLandmarks):
    """perform scale"""
    def __init__(self) -> None:
        image_scale = image.Scale()
        super(Scale, self).__init__(image_scale)
    def _oper(
        self,
        landmarks_wrapper: LandmarksWrapper,
        image_size: ImageSize,
        scale: float
    ) -> LandmarksWrapper:
        landmarks_wrapper.scale(scale)

        return landmarks_wrapper


class Show(object):
    """show image using opencv"""
    def __call__(self, sample: Sample) -> Sample:
        show_landmarks(sample["image"], sample["landmarks"])
        return sample


class ToTensor(ProcLandmarks):
    """convert ndarrays in sample to Tensors."""
    def __init__(self, *args, **kwargs) -> None:
        image_tensor = image.ToTensor(*args, **kwargs)
        super(ToTensor, self).__init__(image_tensor)

    def _oper(
        self,
        landmarks_wrapper: LandmarksWrapper,
        image_size: ImageSize
    ) -> LandmarksWrapper:

        landmarks_wrapper.landmarks = torch.from_numpy(
            landmarks_wrapper.landmarks).type(torch.FloatTensor)

        return landmarks_wrapper


class Normalize(ProcLandmarks):
    """normalization with a give mean and std (R, G, B)"""
    def __init__(self, *args, **kwargs) -> None:
        image_normalize = image.Normalize(*args, **kwargs)
        super(Normalize, self).__init__(image_normalize)


class RandomBlur(RandomTransformer):
    """flip an input image and landmarks horizontally with a given probability
    """
    def __init__(self, prob: float=0.2) -> None:
        super(RandomBlur, self).__init__(prob)
        blur = Blur()
        self.add_op(blur)


class RandomHorizontalFlip(RandomTransformer):
    """flip an input image and landmarks horizontally with a given probability
    """
    def __init__(self, prob: float=0.5) -> None:
        super(RandomHorizontalFlip, self).__init__(prob)
        flip = HorizontalFlip()
        self.add_op(flip)


class RandomCrop_(RandomCropTransformer):
    """make a random crop from the source image and
       the corresponding transformation of landmarks
    """
    def __init__(self, output_size: ImageSize) -> None:
        super(RandomCrop_, self).__init__(output_size)
        crop = Crop()
        self.add_op(crop)


class RandomCrop(RandomTransformer):
    """make a random crop from the source image with a given probability
       perform the same transformation with landmarks
    """
    def __init__(self, output_size: ImageSize, prob: float=0.3) -> None:
        super(RandomCrop, self).__init__(prob)
        crop = RandomCrop_(output_size)
        self.add_op(crop)


class RandomRotate_(RandomRotateTransformer):
    """rotate an image around its center by a random angle
       perform the same transformation with landmarks
    """
    def __init__(self, max_angle: float) -> None:
        super(RandomRotate_, self).__init__(max_angle)
        rotate = Rotate()
        self.add_op(rotate)


class RandomRotate(RandomTransformer):
    """rotate an image around its center by a random angle
       with a given probability
       perform the same transformation with landmarks
    """
    def __init__(self, max_angle: float=10, prob: float=0.1) -> None:
        super(RandomRotate, self).__init__(prob)
        rotate = RandomRotate_(max_angle)
        self.add_op(rotate)


class RandomScale_(RandomScaleTransformer):
    """perform uniform scale by a random magnitude
    """
    def __init__(self, min_scale: float, max_scale: float) -> None:
        super(RandomScale_, self).__init__(min_scale, max_scale)
        scale = Scale()
        self.add_op(scale)


class RandomScale(RandomTransformer):
    """perform uniform scale by a random magnitude with a given prob
    """
    def __init__(
        self,
        min_scale: float=0.8,
        max_scale: float=1.2,
        prob: float=0.3
    ) -> None:
        super(RandomScale, self).__init__(prob)
        scale = RandomScale_(min_scale, max_scale)
        self.add_op(scale)


if __name__ == "__main__":

    from utils.dataset import load_ibug_300w

    import pickle
    import os

    ibug_300w = load_ibug_300w()

    # idx = 0

    # sample = ibug_300w["train"][idx]
    # show_landmarks(sample["image"], sample["landmarks"])

    # # flip
    # flip = HorizontalFlip()
    # sample = flip(sample)
    # show_landmarks(sample["image"], sample["landmarks"])
    # print(sample["image"].shape)

    # # crop
    # crop_roi=(50, 50, 549, 649)
    # crop = Crop()
    # sample = crop(sample, roi=crop_roi)
    # show_landmarks(sample["image"], sample["landmarks"])

    # # rotate
    # angle = -5
    # rot = Rotate()
    # sample = rot(sample, angle=angle)
    # show_landmarks(sample["image"], sample["landmarks"])

    # # scale
    # scale_factor = 0.8
    # scale = Scale()
    # sample = scale(sample, scale=scale_factor)
    # show_landmarks(sample["image"], sample["landmarks"])

    # # resize
    # size = (300, 400)
    # resize = Resize(size)
    # sample = resize(sample)
    # show_landmarks(sample["image"], sample["landmarks"])

    # # to tensor
    # to_tensor = ToTensor()
    # sample = to_tensor(sample)

    # random
    sample = ibug_300w["train"][2]
    show_landmarks(sample["image"], sample["landmarks"])

    size = (500, 500)
    resize = Resize(size)
    sample = resize(sample)
    show_landmarks(sample["image"], sample["landmarks"])

    random_crop = RandomCrop((450, 450))
    sample = random_crop(sample)
    show_landmarks(sample["image"], sample["landmarks"])

    random_rot = RandomRotate()
    sample = random_rot(sample)
    show_landmarks(sample["image"], sample["landmarks"])

    random_scale = RandomScale()
    sample = random_scale(sample)
    show_landmarks(sample["image"], sample["landmarks"])
