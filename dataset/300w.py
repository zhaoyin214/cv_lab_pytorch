from torch.utils.data import Dataset
from typing import Dict, Text
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

from common.define import ImageSize, Landmarks, Transformer
from utils.landmark import norm_01

class IBug300W(Dataset):

    def __init__(
        self, root: Text, label_filepath: Text, transform: Transformer=None,
        padding: float=0.4
    ) -> None:
        self._root = root
        self._transform = transform
        self._padding = padding

        self._image_list = []
        self._box_list = []
        self._landmark_list = []

        self._xml_parser(label_filepath)

    def __getitem__(self, index: int) -> Dict:
        sample = None

        image_filepath = os.path.join(self._root, self._image_list[index])
        image = cv2.imread(image_filepath)
        box = self._box_list[index]
        image = image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
        landmarks = self._landmark_list[index]

        sample ={
            "image": image,
            "landmarks": landmarks
        }

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return len(self._image_list)

    def _xml_parser(self, filepath: Text) -> None:
        """boxes, landmarks

        box: [xmin, ymin, xmax, ymax]
        landmarks: [x0, y0, x1, y1, ..., x67, y67]
        """

        tree = ET.parse(os.path.join(self._root, filepath))
        images = tree.getroot().find("images")

        for image in images.iter("image"):
            self._image_list.append(image.get("file"))
            box = image.find("box")
            box = [
                box.get("left"), box.get("top"),
                box.get("width"), box.get("height")
            ]
            box = [float(item) for item in box]
            # padding
            box[0] -= box[2] * self._padding / 2
            box[1] -= box[3] * self._padding / 2
            box[2] *= 1 + self._padding
            box[3] *= 1 + self._padding
            box[2] += box[0] - 1
            box[3] += box[1] - 1

            landmarks = []
            for landmark in image.iter("part"):
                landmarks.extend([
                    float(landmark.get("x")), float(landmark.get("y"))
                ])

            box[0] = int(min(np.min(landmarks[ : : 2]), box[0]))
            box[2] = int(max(np.max(landmarks[ : : 2]), box[2]))
            box[1] = int(min(np.min(landmarks[1 : : 2]), box[1]))
            box[3] = int(max(np.max(landmarks[1 : : 2]), box[3]))
            self._box_list.append(box)

            for idx in range(len(landmarks) // 2):
                landmarks[2 * idx] -= box[0]
                landmarks[2 * idx + 1] -= box[1]

            w = box[2] - box[0] + 1
            h = box[3] - box[1] + 1
            landmarks = norm_01(ImageSize=(h, w), landmarks=landmarks)

            self._landmark_list.append(landmarks)


if __name__ == "__main__":

    from utils.visual import show_landmarks
    dataset = IBug300W(
        root="d:/proj/datasets/300w",
        label_filepath="labels_ibug_300W_train.xml"
    )

    sample = dataset[0]
    show_landmarks(sample["image"], sample["landmarks"])
    print(sample)