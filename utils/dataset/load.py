from torch.utils.data import Dataset
from typing import List, Text, Union
from copy import deepcopy
import pickle
import os

from config.dataset import IBUG_300W_CONFIG

def load_ibug_300w(reload: bool=False) -> List[Dataset]:

    for phase in ["train", "test"]:
        filepath = IBUG_300W_CONFIG.get(phase)
        if (not os.path.isfile(filepath)) or reload:
            _load_ibug_300w(phase)

    with open(IBUG_300W_CONFIG.get("train"), mode="rb") as f:
        train = pickle.load(f)
    with open(IBUG_300W_CONFIG.get("val"), mode="rb") as f:
        val = pickle.load(f)
    with open(IBUG_300W_CONFIG.get("test"), mode="rb") as f:
        test = pickle.load(f)

    return {"train": train, "val": val, "test": test}


def _load_ibug_300w(phase: Union["train", "test"]) -> None:

    from dataset import IBug300W
    ibug_300w = IBug300W(
        root=IBUG_300W_CONFIG.get("root"),
        label_filepath=IBUG_300W_CONFIG.get("label_" + phase)
    )
    if phase == "train":
        indices_train = [idx for idx in range(len(ibug_300w)) if idx % 3 != 0]
        ibug_300w_train = deepcopy(ibug_300w)
        ibug_300w_train._image_list = []
        ibug_300w_train._bbox_list = []
        ibug_300w_train._landmark_list = []

        for index in indices_train:
            ibug_300w_train._image_list.append(ibug_300w._image_list[index])
            ibug_300w_train._bbox_list.append(ibug_300w._bbox_list[index])
            ibug_300w_train._landmark_list.append(ibug_300w._landmark_list[index])

        with open(IBUG_300W_CONFIG.get("train"), mode="wb") as f:
            pickle.dump(ibug_300w_train, f)

        indices_val = [idx for idx in range(len(ibug_300w)) if idx % 3 == 0]
        ibug_300w_val = deepcopy(ibug_300w)
        ibug_300w_val._image_list = []
        ibug_300w_val._bbox_list = []
        ibug_300w_val._landmark_list = []

        for index in indices_val:
            ibug_300w_val._image_list.append(ibug_300w._image_list[index])
            ibug_300w_val._bbox_list.append(ibug_300w._bbox_list[index])
            ibug_300w_val._landmark_list.append(ibug_300w._landmark_list[index])

        with open(IBUG_300W_CONFIG.get("val"), mode="wb") as f:
            pickle.dump(ibug_300w_val, f)
    else:
        with open(IBUG_300W_CONFIG.get("test"), mode="wb") as f:
            pickle.dump(ibug_300w, f)
