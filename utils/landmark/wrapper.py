import cv2
import numpy as np

from common.define import Box, Landmarks, Point

class LandmarksWrapper(object):

    def __init__(self, landmarks: Landmarks=None) -> None:
        self.landmarks = landmarks

    def __getitem__(self, index: int) -> Point:
        return self._landmarks[2 * index], self._landmarks[2 * index + 1]

    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks) // 2

    @property
    def xs(self) -> np.ndarray:
        return self._landmarks[: : 2]
    @xs.setter
    def xs(self, value: np.ndarray) -> None:
        self._landmarks[: : 2] = value

    @property
    def ys(self) -> np.ndarray:
        return self._landmarks[1 : : 2]
    @ys.setter
    def ys(self, value: np.ndarray) -> None:
        self._landmarks[1 : : 2] = value

    @property
    def landmarks(self) -> np.ndarray:
        return self._landmarks
    @landmarks.setter
    def landmarks(self, value: Landmarks) -> None:
        self._landmarks = value

    def horizontal_shift(self, value: float) -> None:
        self.xs += value

    def vertical_shift(self, value: float) -> None:
        self.ys += value

    def horizontal_flip(self) -> None:
        self.xs = 1 - self.xs

    def vertical_flip(self) -> None:
        self.ys = 1 - self.ys

    def horizontal_scale(self, value: float) -> None:
        self.xs *= value

    def vertical_scale(self, value: float) -> None:
        self.ys *= value

    def rotate(self, value: float) -> None:
        rot_mat = cv2.getRotationMatrix2D(
            center=(0.5, 0.5), angle=value, scale=1
        )
        self._transform(rot_mat)

    def scale(self, value: float) -> None:
        scale_mat = cv2.getRotationMatrix2D(
            center=(0.5, 0.5), angle=0, scale=value
        )
        self._transform(scale_mat)

    def _transform(self, mat) -> None:
        landmarks = np.array([self.xs, self.ys, np.ones(shape=(68, ))]).T
        landmarks = landmarks.reshape(1, -1, 3)
        landmarks = cv2.transform(landmarks, mat)
        self.xs = landmarks[0, :, 0]
        self.ys = landmarks[0, :, 1]

    def norm_01(self, box: Box) -> None:

        self.horizontal_shift(-box[0])
        self.vertical_shift(-box[1])
        w = box[2] - box[0] + 1
        h = box[3] - box[1] + 1
        self.horizontal_scale(1 / w)
        self.vertical_scale(1 / h)