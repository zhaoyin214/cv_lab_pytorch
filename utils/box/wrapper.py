import numpy as np
from common.define import Box, ImageSize, Point

class BoxWrapper(object):

    def __init__(self, box: Box=None) -> None:
        if box:
            self.box = box.astype(np.int)
        else:
            self.box = np.empty(shape=(4, ), dtype=np.int)

    @property
    def xmin(self) -> int:
        return self._box[0]
    @xmin.setter
    def xmin(self, value: int) -> None:
        self._box[0] = value

    @property
    def xmax(self) -> int:
        return self._box[2]
    @xmax.setter
    def xmax(self, value: int) -> None:
        self._box[2] = value

    @property
    def ymin(self) -> int:
        return self._box[1]
    @ymin.setter
    def ymin(self, value: int) -> None:
        self._box[1] = value

    @property
    def ymax(self) -> int:
        return self._box[3]
    @ymax.setter
    def ymax(self, value: int) -> None:
        self._box[3] = value

    @property
    def left_top_corner(self) -> Point:
        return self.xmin, self.ymin

    @property
    def right_bottom_corner(self) -> Point:
        return self.xmax, self.ymax

    @property
    def width(self) -> int:
        return self.xmax - self.xmin + 1

    @property
    def height(self) -> int:
        return self.ymax - self.ymin + 1

    @property
    def box(self) -> Box:
        return self._box
    @box.setter
    def box(self, value: Box) -> None:
        self._box = value.astype(np.int)

    def set_box_ltwh(
        self, left: int, top: int, width: int, height: int
    ) -> None:
        self.xmin = int(left)
        self.ymin = int(top)
        self.xmax = int(left + width - 1)
        self.ymax = int(top + height - 1)


    def to_box_ltwh(self):
        return np.array([
            self.xmin, self.ymin,
            self.xmax - self.xmin + 1, self.ymax - self.ymin + 1
        ])


class BBoxWrapper(object):

    def __init__(self, bbox: Box, image_size: ImageSize) -> None:
        self._box_wrapper = BoxWrapper(bbox)
        self._w, self._h = image_size