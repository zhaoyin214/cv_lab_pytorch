from typing import Dict, Tuple, Union, List
from torchvision.transforms import Compose
import numpy as np

ImageSize = Union[int, Tuple]   # [width, height]
Image = np.ndarray              # image, 2-dim or 3-dim array
Transformer = Compose
Landmarks = np.ndarray          # 1-dim array, [x0, y0, x1, y1, ...]
Sample = Dict                   # {"image": image, "landmarks": landmarks, "bbox": bbox}

Point = List[int]               # [x, y]