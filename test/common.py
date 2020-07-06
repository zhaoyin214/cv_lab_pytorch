from typing import List, Text, Tuple, Union
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import numpy as np
import os
import torch

class BaseTester(object):
    """base tester for visualization

    arguments:

    filepath: a trained network
    """

    def __init__(
        self,
        model: Module=None,
        dataset: Dataset=None,
        classes: List[Text]=None,
        num_workers: int=0,
        batch_size: int=16,
        device: Text="gpu",
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

        if classes:
            self.set_classes(classes)
        else:
            self._classes = None
            self._id2class = None

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
            self._logger.warning("warning: device {} is not available".format(device))
        if device == "gpu" and torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

    def set_batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._num_grid_rows = int(np.ceil(np.sqrt(self._batch_size)))

    def set_classes(self, classes: List[Text]) -> None:
        self._classes = classes
        self._id2class = {
            idx: term for idx, term in enumerate(classes)
        }

    def __call__(self, x: torch.Tensor):
        preds = self._model(x)
        return preds

    def _post_proc(
        self,
        x: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[np.ndarray, List[Text], List[Text]]:

        x = self._make_grid(x)
        preds = torch.argmax(preds, dim=1)
        preds = [
            self._id2class[int(pred)] for pred in preds
        ]
        labels = [
            self._id2class[int(label)] for label in labels
        ]
        return x, preds, labels

    def _make_grid(self, x: torch.Tensor) -> np.ndarray:

        x = make_grid(x, nrow=self._num_grid_rows, padding=0)
        x = x.cpu().numpy().transpose((1, 2, 0))
        x = self._std * x + self._mean
        x = np.clip(x, 0, 1)

        return x

    def _fetch_input_label(self, samples):
        return samples

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

if __name__ == "__main__":

    from torchvision import datasets, transforms
    from torchvision.models import resnet18, resnet34
    from torch.optim.lr_scheduler import StepLR
    import cv2
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

    tester = BaseTester(
        net, dataset, classes
    )

    for image_grid, preds, labels in tester.batch_pred():
        image_grid *= 255
        image_grid = image_grid.astype(np.uint8)
        image_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)
        cv2.imshow("grid", image_grid)
        cv2.waitKey(20)

    cv2.destroyAllWindows()
