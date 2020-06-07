from typing import Dict, Tuple, Union, List
from torchvision.transforms import Compose
import numpy as np

IMAGE_SIZE = Union[int, Tuple]  # [height, width]
TRANSFORM = Compose
LANDMARKS = np.ndarray          # 1-dim array, [x0, y0, x1, y1, ...]