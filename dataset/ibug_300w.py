from typing import Text
import numpy as np
import os
import xml.etree.ElementTree as ET

from .landmark import LandmarkDataset

class IBug300W(LandmarkDataset):

    def _parser(self, filepath: Text) -> None:
        """bboxes, landmarks

        bbox: [xmin, ymin, xmax, ymax]
        landmarks: [x0, y0, x1, y1, ..., x67, y67]
        """

        tree = ET.parse(os.path.join(self._root, filepath))
        images = tree.getroot().find("images")

        for image in images.iter("image"):
            self._image_list.append(image.get("file"))
            bbox = image.find("box")
            bbox = [
                bbox.get("left"), bbox.get("top"),
                bbox.get("width"), bbox.get("height")
            ]
            bbox = np.array([float(item) for item in bbox])
            self._bbox_list.append(bbox)

            landmarks = []
            for landmark in image.iter("part"):
                landmarks.extend([
                    float(landmark.get("x")), float(landmark.get("y"))
                ])
            landmarks = np.array(landmarks)
            self._landmark_list.append(landmarks)


if __name__ == "__main__":

    from utils.visual import show_landmarks
    from utils.dataset import load_ibug_300w

    dataset = load_ibug_300w(True)["train"]
    sample = dataset[0]
    show_landmarks(sample["image"], sample["landmarks"])

    from torchvision.transforms import Compose
    from transform.landmark import RandomBlur, RandomCrop, \
        RandomHorizontalFlip, Resize, ToTensor, Normalize, \
        RandomRotate, RandomScale, Show

    transform = Compose([
        Resize((500, 500)),
        RandomRotate(max_angle=30),
        RandomBlur(),
        RandomHorizontalFlip(),
        RandomCrop((450, 450)),
        RandomScale(),
        Resize((450, 450)),
        Show(),
        ToTensor(),
        Normalize()
    ])

    dataset.transform = transform

    for idx in range(10):
        sample = dataset[idx]
        sample = dataset[idx]