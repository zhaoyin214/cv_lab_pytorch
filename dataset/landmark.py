from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
from typing import List, Text
import cv2
import numpy as np
import os

from common.define import Box, ImageSize, Landmarks, Sample, Transformer
from utils.landmark import LandmarksWrapper
from utils.image import crop

class LandmarkDataset(Dataset, metaclass=ABCMeta):

    def __init__(
        self, root: Text, label_filepath: Text, transform: Transformer=None,
        expanding: float=0.5
    ) -> None:
        self._root = root
        self._transform = transform
        self._expanding = expanding

        self._image_list = []
        self._bbox_list = []
        self._landmark_list = []

        self._parser(label_filepath)

    def __getitem__(self, index: int) -> Sample:

        # image
        image_filepath = os.path.join(self._root, self._image_list[index])
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[: 2]

        # roi
        bbox = self._bbox_list[index].copy()
        # expanding
        bbox = self._expand(bbox, (w, h))
        # cropping roi
        image = crop(image, bbox)

        # landmarks
        landmarks = self._landmark_list[index].copy()
        landmarks_wrapper = LandmarksWrapper(landmarks)
        # 0-1 normalization
        landmarks_wrapper.norm_01(bbox)

        sample ={
            "image": image,
            "landmarks": landmarks_wrapper.landmarks
        }

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return len(self._image_list)

    @abstractmethod
    def _parser(self, filepath: Text) -> None:
        """bboxes, landmarks

        bbox: [xmin, ymin, xmax, ymax]
        landmarks: [x0, y0, x1, y1, ..., x67, y67]
        """
        pass

    def _expand(self, bbox: Box, image_size: ImageSize) -> Box:

        w, h = image_size
        # expanding
        bbox[0] -= bbox[2] * self._expanding / 2
        bbox[1] -= bbox[3] * self._expanding / 2
        bbox[2] *= 1 + self._expanding
        bbox[3] *= 1 + self._expanding
        bbox[2] += bbox[0] - 1
        bbox[3] += bbox[1] - 1

        bbox[0] = max(0, bbox[0])
        bbox[2] = min(w, bbox[2])
        bbox[1] = max(0, bbox[1])
        bbox[3] = min(h, bbox[3])
        bbox = bbox.astype(np.int)

        return bbox

    @property
    def expanding(self) -> float:
        return self._expanding
    @expanding.setter
    def expanding(self, value: float) -> None:
        self._expanding = value

    @property
    def image_list(self) -> List:
        return self._image_list

    @property
    def bbox_list(self) -> List:
        return self._bbox_list

    @property
    def landmark_list(self) -> List:
        return self._landmark_list

    @property
    def transform(self) -> Transformer:
        return self._transform
    @transform.setter
    def transform(self, value: Transformer) -> None:
        self._transform = value