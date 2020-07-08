from abc import ABCMeta, abstractmethod
from typing import List, Text, Tuple, Union
import cv2
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import torch

class AbstractTester(metaclass=ABCMeta):
    """abstract tester for visualization

    arguments:

    filepath: a trained network
    """

    def __init__(
        self,
        model: Module=None,
        dataset: Dataset=None,
        num_workers: int=0,
        batch_size: int=16,
        device: Text="cpu",
        mean: Union[int, List]=[0.485, 0.456, 0.406],
        std: Union[int, List]=[0.229, 0.224, 0.225]
    ) -> None:

        self.set_device(device)
        self._mean = mean
        self._std = std
        self._num_workers = num_workers
        self.set_batch_size(batch_size)

        # model
        self.set_model(model)
        # dataset
        if dataset:
            self.set_dataset(dataset)
        else:
            self._dataset = None

    def set_dataset(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers
        )
        self._dataset_size = len(dataset)

    def set_model(self, model: Module) -> None:
        self._model = model.to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

    def set_device(self, device: Text) -> None:
        assert device in ["cpu", "gpu"], \
            "warning: device {} is not available".format(device)
        if device == "gpu" and torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

    def set_batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._num_grid_rows = int(np.ceil(np.sqrt(self._batch_size)))

    def __call__(self, x: torch.Tensor):
        preds = self._model(x)
        return preds

    def _fetch_input_label(self, samples):
        return samples

    def _post_proc(
        self,
        x: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple:
        return self._proc_batch(x, preds, labels)

    def _proc_batch(
        self,
        x: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple:
        x = self._proc_image(x)
        preds = self._proc_pred(preds)
        labels = self._proc_label(labels)
        # label images
        for idx in range(x.shape[0]):
            x[idx, :, :, :] = self._extra_proc_image(
                x[idx, :, :, :],
                preds[idx],
                labels[idx]
            )
        # array -> tensor, with shape (b, c, h, w)
        x = x.transpose(0, 3, 1, 2)
        x = torch.from_numpy(x)
        # make a grid
        x = self._make_grid(x)

        return x, preds, labels

    def _proc_image(self, x) -> np.ndarray:
        """tensor -> array, with shape (b, h, w, c)"""
        x = x.cpu().numpy()
        x = x.transpose(0, 2, 3, 1)
        x = self._std * x + self._mean
        x = np.clip(x, 0, 1)
        x *= 255
        return x.astype(np.uint8)

    def _extra_proc_image(
        self, image: np.ndarray, pred, label,
    ) -> np.ndarray:
        return image

    @abstractmethod
    def _proc_pred(self, preds: torch.Tensor) -> List:
        pass

    @abstractmethod
    def _proc_label(self, labels: torch.Tensor) -> List:
        pass

    def _make_grid(self, x: torch.Tensor) -> np.ndarray:
        x = make_grid(x, nrow=self._num_grid_rows, padding=0)
        x = x.cpu().numpy().transpose((1, 2, 0))
        x = x.astype(np.uint8)
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    def batch_pred(self) -> np.ndarray:

        iter_dataloader = iter(self._data_loader)
        while True:
            try:
                samples = next(iter_dataloader)
                images, labels = self._fetch_input_label(samples)
                images = images.to(self._device)
                preds = self._model(images)
                images, preds, labels = self._post_proc(images, preds, labels)
                yield images, preds, labels
            except StopIteration as e:
                raise e

class ClassTester(AbstractTester):

    def set_classes(self, classes: List[Text]) -> None:
        self._classes = classes
        self._id2class = {
            idx: term for idx, term in enumerate(classes)
        }

    def _extra_proc_image(
        self,
        image: np.ndarray,
        pred: Text,
        label: Text,
    ) -> np.ndarray:
        text = "pred: {}, gt: {}".format(pred, label)
        image = image.copy()
        image = cv2.putText(
            img=image, text=text, org=(10, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        return image

    def _proc_pred(self, preds: torch.Tensor) -> List:
        """prediction, id -> class"""
        preds = torch.argmax(preds, dim=1)
        preds = [
            self._id2class[int(pred)] for pred in preds
        ]
        return preds

    def _proc_label(self, labels: torch.Tensor) -> List:
        """label, id -> class"""
        labels = [
            self._id2class[int(label)] for label in labels
        ]
        return labels

if __name__ == "__main__":

    from torchvision import datasets, transforms
    from torchvision.models import resnet18, resnet34
    from torch.optim.lr_scheduler import StepLR
    import os

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    filepath = "./output/model/best_resnet34-epoch_12-val_acc_0.9412-val_loss_0.3704.pth"
    net = resnet34(pretrained=True)
    in_planes = net.fc.in_features
    net.fc = torch.nn.Linear(in_planes, 2)
    net.load_state_dict(torch.load(filepath))

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    root = r"D:\proj\datasets\hymenoptera_data"
    dataset = datasets.ImageFolder(
        root=os.path.join(root, "val"),
        transform=data_transform
    )
    classes = ["ant", "bee"]

    tester = ClassTester(net, dataset)
    tester.set_classes(classes)

    for image_grid, preds, labels in tester.batch_pred():
        cv2.imshow("grid", image_grid)
        cv2.waitKey(2000)

    cv2.destroyAllWindows()
