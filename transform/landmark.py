#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   transform.py
@time    :   2020/06/05 18:59:57
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :
"""

__author__ = "XiaoY"


import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Union

from common.define import IMAGE_SIZE
from .common import LANDMARK_SAMPLE

__all__ = [
    "Blur", "HorizontalFlip", "RandomCrop", "RandomRotate", "RandomScale", "Rescale", "ToTensor"
]

class Random(object):
    """"uniform-randomly choice with a given probability"""
    def __init__(self, prob: float=0.5) -> None:
        self._prob = prob

    def _choice(self) -> bool:
        return float(np.random.rand(1)) < self._prob


class Blur(Random):
    """blurs an image with the given sigma and probability

    arguments:
        prob: the probability to blur the image
        ksize: kernel size
    """
    def __init__(self, prob: float=0.3, ksize: int=3) -> None:
        super(Blur, self).__init__(prob)
        assert ksize % 2 == 1
        self._ksize = ksize

    def __call__(self, sample: Dict) -> Dict:
        image, landmarks = sample["image"], sample["landmarks"]

        if self._choice():
            image = cv2.blur(image, (self._ksize, self._ksize))

        return {"image": image, "landmarks": landmarks}


class HorizontalFlip(Random):
    """Flips an input image and landmarks horizontally with a given probability"""

    def __call__(self, sample: LANDMARK_SAMPLE) -> LANDMARK_SAMPLE:
        image, landmarks = sample["image"], sample["landmarks"].reshape(-1, 2)

        if self._choice():
            image = cv2.flip(image, 1)
            landmarks = landmarks.reshape(-1, 2)
            landmarks[:, 0] = 1. - landmarks[:, 0]
            tmp = np.copy(landmarks[0])
            landmarks[0] = landmarks[1]
            landmarks[1] = tmp

            tmp = np.copy(landmarks[3])
            landmarks[3] = landmarks[4]
            landmarks[4] = tmp

        return {"image": image, "landmarks": landmarks}


class RandomCrop:
    """Makes a random crop from the source image with corresponding transformation of landmarks"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self._output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self._output_size = output_size

    def __call__(self, sample: LANDMARK_SAMPLE) -> LANDMARK_SAMPLE:
        image, landmarks = sample["image"], sample["landmarks"].reshape(-1, 2)

        h, w = image.shape[:2]
        new_h, new_w = self._output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left / float(w), top / float(h)]
        for point in landmarks:
            point[0] *= float(h) / new_h
            point[1] *= float(w) / new_w

        return {"image": image, "landmarks": landmarks}


class RandomRotate:
    """
        Rotates an image around it"s center by a randomly generated angle.
        Also performs the same transformation with landmark points.
    """
    def __init__(self, max_angle, p=.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, sample: LANDMARK_SAMPLE) -> LANDMARK_SAMPLE:
        image, landmarks = sample["image"], sample["landmarks"]

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

    def __call__(self, sample: LANDMARK_SAMPLE) -> LANDMARK_SAMPLE:
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
    def __init__(self, output_size: IMAGE_SIZE) -> None:
        self._output_size = output_size

    def __call__(self, sample: LANDMARK_SAMPLE) -> LANDMARK_SAMPLE:
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
    def __call__(self, sample: LANDMARK_SAMPLE) -> LANDMARK_SAMPLE:
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

    def __call__(self, sample: LANDMARK_SAMPLE) -> LANDMARK_SAMPLE:
        image, landmarks = sample["image"], sample["landmarks"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.switch_rb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image).type(torch.FloatTensor) / 255,
                "landmarks": torch.from_numpy(landmarks).type(torch.FloatTensor).view(-1, 1, 1)}

