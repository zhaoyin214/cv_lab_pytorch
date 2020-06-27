from abc import ABCMeta, abstractmethod
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Text
import copy
import os
import time

from utils.log import Logger
from .metric import MetricManager


class BaseTrainer(object):
    """base trainer

    arguments:

    model: network
    optimizer: adam, sgd, rmsprop, ...
    criterion: loss
    scheduler: steplr, ...
    log_file:
    """

    _phase_list = ["train", "val"]
    def __init__(
        self,
        model: Module=None,
        optimizer: Optimizer=None,
        criterion: Module=None,
        scheduler: LRScheduler=None,
        datasets: Dict[Text, Dataset]=None,
        metric_manager: MetricManager=None,
        model_name: Text="",
        log_filename: Text="training_log",
        save_dir: Text="./output",
        num_workers: int=2,
        batch_size: int=256,
        num_epochs: int=1000,
        device: Text="gpu"
    ) -> None:

        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._metric_manager = metric_manager
        self._model_name = model_name
        self._save_dir = save_dir
        log_filename = os.path.join(
            self._save_dir, "log/{}_{}_{}.log".format(
                model_name,
                log_filename,
                time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            )
        )
        self._logger = Logger(log_filename, level="info")

        self._num_workers = num_workers
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self.device = device
        self._is_test = False

        if datasets:
            self.set_dataset(datasets)
        else:
            self._datasets = None

    def _set_device(self, device: Text) -> None:
        assert device in ["cpu", "gpu"], \
            self._logger.warning("warning: device {} is not available".format(device))
        if device == "gpu" and torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

    def set_model(self, model: Module) -> None:
        self._model = model

    def set_criterion(self, criterion: Module) -> None:
        self._criterion = criterion

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def set_lr_scheduler(self, scheduler: LRScheduler) -> None:
        self._scheduler = scheduler

    def set_dataset(self, datasets: Dict[Text, Dataset]) -> None:
        self._datasets = datasets

        # training and val
        self._data_loaders = {
            x: torch.utils.data.DataLoader(
                dataset=datasets[x],
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=self._num_workers
            ) for x in self._phase_list
        }

        # test
        if datasets.get("test"):
            self._data_loaders["test"] = torch.utils.data.DataLoader(
                dataset=datasets["test"],
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers
            )
            self._is_test = True

    def set_metric_manager(self, metric_manager: MetricManager) -> None:
        self._metric_manager = metric_manager

    @property
    def num_epochs(self) -> int:
        return self._num_epoches
    @num_epochs.setter
    def num_epochs(self, value: int) -> None:
        self._num_epoches = value

    @property
    def batch_size(self) -> int:
        return self._batch_size
    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = value

    @property
    def device(self) -> Text:
        return self._device
    @device.setter
    def device(self, device: Text) -> None:
        self._set_device(device)


    def _check(self) -> None:
        assert isinstance(self._model, Module), \
            self._logger.error("error: the net is not availabe.")
        assert isinstance(self._criterion, Module), \
            self._logger.error("error: the loss is not availabe.")
        assert isinstance(self._optimizer, Optimizer), \
            self._logger.error("error: the optimizer is not availabe.")
        assert isinstance(self._datasets["train"], Dataset), \
            self._logger.error("error: the train dataset is not availabe.")
        assert isinstance(self._datasets["val"], Dataset), \
            self._logger.error("error: the val dataset is not availabe.")

    def _fetch_input_label(self, samples):
        return samples

    def train(self) -> Module:
        """train and validation
        only best model saved in validation phases
        """

        self._check()
        self._model.to(self.device)

        since = time.time()

        best_model_wts = copy.deepcopy(self._model.state_dict())
        self._metric_manager.clear_best_metric()

        for epoch in range(self._num_epochs):

            self._logger.info("-" * 30)
            self._logger.info("epoch {}/{}".format(epoch, self._num_epochs - 1))

            # each epoch has a training and validation phase
            for phase in self._phase_list:

                epoch_stats = self._run_iter(phase)

                log_info = "[{}] ".format(phase)
                for key, value in epoch_stats.items():
                    log_info += "{}: {:.4f}, ".format(key, value)
                self._logger.info(log_info)

                # deep copy the model
                if phase == "val":
                    if self._metric_manager.is_best():
                        best_model_wts = copy.deepcopy(self._model.state_dict())
                        log_info = ""
                        for key, value in epoch_stats.items():
                            log_info += "-val_{}_{:.4f}".format(key, value)
                        torch.save(
                            self._model.state_dict(),
                            os.path.join(
                                self._save_dir, "model",
                                "best_{}-epoch_{}{}.pth".format(
                                    self._model_name if self._model_name else "model",
                                    epoch, log_info
                                )
                            )
                        )

        time_elapsed = time.time() - since
        self._logger.info("-" * 30)
        self._logger.info("training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))
        self._logger.info("best val {}: {:4f}".format(
            self._metric_manager.metric_name, self._metric_manager.best_metric))

        # load best self._model weights
        self._model.load_state_dict(best_model_wts)

        return self._model

    def test(self) -> None:
        """test
        being called only once after training
        """

        assert self._is_test, ValueError("error: test dataset is not available.")

        self._logger.info("-" * 30)
        self._logger.info("-- test phase --")
        phase = "test"
        since = time.time()

        epoch_stats = self._run_iter(phase)
        log_info = "[{}] ".format(phase)
        for key, value in epoch_stats.items():
            log_info += "{}: {:.4f}, ".format(key, value)
        self._logger.info(log_info)

        time_elapsed = time.time() - since
        self._logger.info("test complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))

        return None

    def _run_iter(self, phase: Text) -> Dict[Text, float]:

        dataset_sizes = {x: len(self._datasets[x]) for x in self._datasets.keys()}

        if phase == "train":
            # set model to training mode
            self._model.train()
        else:
            # set model to evaluate mode
            self._model.eval()

        self._metric_manager.zero_stats()

        # iterate over data
        for samples in self._data_loaders[phase]:
            inputs, labels = self._fetch_input_label(samples)
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            # zero the parameter gradients
            self._optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = self._model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self._criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    self._optimizer.step()
                    if self._scheduler:
                        self._scheduler.step()

            # statistics
            self._metric_manager.running_call(preds, labels, loss)

        epoch_stats = self._metric_manager.epoch_call(dataset_sizes[phase], phase)

        return epoch_stats


if __name__ == "__main__":

    from torchvision import datasets, transforms
    from torchvision.models import resnet18, resnet34
    from torch.optim.lr_scheduler import StepLR
    import os

    from .metric import MetricAcc, MetricLoss

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    root = r"D:\proj\datasets\hymenoptera_data"
    dataset = {
        x: datasets.ImageFolder(
            root=os.path.join(root, x),
            transform=data_transforms[x]
        ) for x in ["train", "val"]
    }
    dataset["test"] = dataset["val"]

    net = resnet34(pretrained=True)
    in_planes = net.fc.in_features
    net.fc = torch.nn.Linear(in_planes, 2)
    # net[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    exp_lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    metric_manager = MetricManager()
    metric_manager.add_stat("acc", MetricAcc())
    metric_manager.add_stat("loss", MetricLoss())
    metric_manager.set_metric("acc")

    trainer = BaseTrainer(
        net,
        optimizer,
        criterion,
        scheduler=exp_lr_scheduler,
        datasets=dataset,
        metric_manager=metric_manager,
        model_name="resnet34",
        num_epochs=20,
        batch_size=512,
        num_workers=2
    )

    net = trainer.train()
    trainer.test()

    metric_manager.zero_stats()
    logs = metric_manager.logs
    print(logs)
