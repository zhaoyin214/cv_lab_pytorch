from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, List, Text
import sys
import torch

class BaseMetric(metaclass=ABCMeta):
    """the abstract class of metrics
       the sub class must implement the 'running stat' method

    """
    def __init__(self) -> None:
        self._running_stat = 0

    @abstractmethod
    def calc_running_stat(
        self, preds: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
    ) -> None:
        pass

    def calc_epoch_stat(self, size: int):
        return self._running_stat / size

    def zero_stat(self) -> None:
        self._running_stat = 0

    @abstractproperty
    def is_max_obj(self) -> bool:
        pass


class MetricLoss(BaseMetric):
    """statistic loss"""
    def calc_running_stat(
        self, preds: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
    ) -> None:
        self._running_stat += loss.item() * preds.size(0)

    def is_max_obj(self) -> bool:
        return False

class MetricAcc(BaseMetric):
    """statistic accuracy"""
    def calc_running_stat(
        self, preds: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
    ) -> None:
        self._running_stat += torch.sum(preds == labels.data).item()

    def is_max_obj(self) -> bool:
        return True

class MetricManager(object):

    def __init__(self):
        self._stat_map = {}
        self._metric_name = None
        self._best_metric = None
        self._epoch_stats = {}

    def add_stat(self, name: Text, stat: BaseMetric) -> None:
        self._stat_map[name] = stat
        self._epoch_stats[name] = []

    def zero_stats(self) -> None:
        for key in self._stat_map.keys():
            self._stat_map[key].zero_stat()

    def set_metric(self, name: Text) -> None:
        self._metric_name = name

    def clear_best_metric(self) -> None:
        self._best_metric = \
            sys.float_info.min if self.is_max_metric \
            else sys.float_info.max

    def is_best(self) -> bool:

        metric = self._epoch_stats[self._metric_name][-1]
        ret = False
        if self.is_max_metric and (self._best_metric < metric):
            self._best_metric = metric
            ret = True

        if (not self.is_max_metric) and (self._best_metric > metric):
            self._best_metric = metric
            ret = True

        return ret

    def running_call(
        self, preds: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
    ) -> None:
        for key in self._stat_map.keys():
            self._stat_map[key].calc_running_stat(preds, labels, loss)

    def epoch_call(self, size: int) -> Dict[Text, float]:
        ret = {
            key: self._stat_map[key].calc_epoch_stat(size)
            for key in self._stat_map.keys()
        }
        for key in self._stat_map.keys():
            self._epoch_stats[key].append(ret[key])
        return ret

    @property
    def is_max_metric(self) -> bool:
        return self._stat_map[self._metric_name].is_max_obj()

    @property
    def best_metric(self) -> bool:
        return self._best_metric

    @property
    def metric_name(self) -> Text:
        return self._metric_name

    @property
    def epoch_stat(self) -> Dict[Text, List]:
        return self._epoch_stats