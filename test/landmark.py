from typing import Dict
import numpy as np
import torch

from .common import AbstractTester
from utils.visual import draw_landmarks

class LandmarkTester(AbstractTester):

    def set_landmark_config(self):
        pass

    def _fetch_input_label(self, samples: Dict):
        return samples["image"], samples["landmarks"]

    def _extra_proc_image(
        self,
        image: np.ndarray,
        pred: np.ndarray,
        label: np.ndarray,
    ) -> np.ndarray:
        image = draw_landmarks(image, pred)
        # image = draw_landmarks(image, label)
        return image

    def _proc_pred(self, preds: torch.Tensor) -> np.ndarray:
        """prediction, tensor -> array"""
        return preds.cpu().numpy()

    def _proc_label(self, labels: torch.Tensor) -> np.ndarray:
        """label, tensor -> array"""
        return labels.cpu().numpy()

if __name__ == "__main__":
    from torchvision.transforms import Compose
    import cv2

    from dataset import IBug300W
    from model.backbone import BackBoneFactory
    from model.landmark import BaselineNet
    from transform.landmark import Resize, ToTensor, Normalize
    from utils.dataset import load_ibug_300w

    dataset = load_ibug_300w()["test"]

    data_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize()
    ])
    dataset.transform = data_transform

    # filepath = "./output/model/best_landmark_baseline_resnet18-epoch_12-val_loss_0.0034.pth"
    # backbone_ = "resnet18"
    filepath = "./output/model/best_landmark_baseline_resnet34_criterion_smooth_l1-epoch_19-val_loss_0.0013.pth"
    backbone_ = "resnet34"
    # backbone_ = "resnet50"
    # backbone_ = "resnet101"
    backbone = BackBoneFactory()(backbone_)
    net = BaselineNet(backbone)
    net.load_state_dict(torch.load(filepath))

    tester = LandmarkTester(net, dataset)

    for cnt, (image_grid, preds, labels) in enumerate(tester.batch_pred()):
        cv2.imshow("grid", image_grid)
        cv2.waitKey(2000)
        if cnt > 5:
            break

    cv2.destroyAllWindows()
