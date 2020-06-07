from typing import Dict
import cv2
import numpy as np

from common.define import Image, ImageSize
from utils.random import RandomChoice, RandomOrig
from utils.image.size import convert_size

class Blur(object):
    """blurs an image with the given sigma and probability

    arguments:
        prob: the probability to blur the image
        ksize: kernel size
    """
    def __init__(self, prob: float=0.3, ksize: int=3) -> None:
        self._random_choice = RandomChoice(prob)
        assert ksize % 2 == 1
        self._ksize = ksize

    def __call__(self, image: Image) -> Image:

        self._random_choice.update()
        if self._random_choice():
            image = cv2.blur(image, (self._ksize, self._ksize))

        return image


class HorizontalFlip(object):
    """Flips an input image and landmarks horizontally with a given probability"""
    def __init__(self, prob: float=0.5) -> None:
        self._random_choice = RandomChoice(prob)

    def __call__(self, image: Image) -> Image:

        self._random_choice.update()
        if self._random_choice():
            image = cv2.flip(image, 1)

        return image


class RandomCrop(object):
    """Makes a random crop from the source image with corresponding transformation of landmarks"""
    def __init__(self, output_size: ImageSize) -> None:
        self._output_size = convert_size(output_size)
        self._random_orig = RandomOrig()

    def __call__(self, image: Image) -> Image:

        h, w = image.shape[:2]
        new_h, new_w = self._output_size

        self._random_orig.update(x_range=w - new_w, y_range=h - new_h)
        left, top = self._random_orig()

        image = image[
            top : top + new_h, left : left + new_w, ...
        ]

        return image


class RandomRotate(object):
    """Rotates an image around it's center by a randomly generated angle.
    """
    def __init__(self, max_angle, p: float=0.5):
        self._random_angle = ()
        self.max_angle = max_angle
        self.p = p

    def __call__(self, image: Image) -> Image:

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            angle = 2*(torch.FloatTensor(1).uniform_() - .5)*self.max_angle
            h, w = image.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, 1.)
            image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LANCZOS4)
            rot_mat_l = cv2.getRotationMatrix2D((0.5, 0.5), angle, 1.)
            landmarks = cv2.transform(landmarks.reshape(1, 5, 2), rot_mat_l).reshape(5, 2)

        return {"image": image, "landmarks": landmarks}


class RandomScale:
    """Performs uniform scale with a random magnitude"""
    def __init__(self, max_scale, min_scale, p=.5):
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.p = p

    def __call__(self, image: Image) -> Image:
        image, landmarks = sample["image"], sample["landmarks"]

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            scale = self.min_scale + torch.FloatTensor(1).uniform_()*(self.max_scale - self.min_scale)
            h, w = image.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((w*0.5, h*0.5), 0, scale)
            image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LANCZOS4)
            rot_mat_l = cv2.getRotationMatrix2D((0.5, 0.5), 0, scale)
            landmarks = cv2.transform(landmarks.reshape(1, 5, 2), rot_mat_l).reshape(5, 2)

        return {"image": image, "landmarks": landmarks}


class Rescale:
    """resizes an image and corresponding landmarks"""
    def __init__(self, output_size: ImageSize) -> None:
        self._output_size = output_size

    def __call__(self, image: Image) -> Image:
        image, landmarks = sample["image"], sample["landmarks"]

        h, w = image.shape[:2]
        if isinstance(self._output_size, int):
            if w > h:
                new_h, new_w = self._output_size, self._output_size * w / h
            else:
                new_h, new_w = self._output_size * h / w, self._output_size
        else:
            new_h, new_w = self._output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_h, new_w))
        return {"image": img, "landmarks": landmarks}


class Show:
    """Show image using opencv"""
    def __call__(self, image: Image) -> Image:
        image, landmarks = sample["image"].copy(), sample["landmarks"].reshape(-1, 2)
        h, w = image.shape[:2]
        for point in landmarks:
            cv2.circle(image, (int(point[0]*w), int(point[1]*h)), 3, (255, 0, 0), -1)
        cv2.imshow("image", image)
        cv2.waitKey()
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, switch_rb=False):
        self.switch_rb = switch_rb

    def __call__(self, image: Image) -> Image:
        image, landmarks = sample["image"], sample["landmarks"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.switch_rb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image).type(torch.FloatTensor) / 255,
                "landmarks": torch.from_numpy(landmarks).type(torch.FloatTensor).view(-1, 1, 1)}
