from common.define import IMAGE_SIZE

def image_size(input_size: IMAGE_SIZE, output_size: IMAGE_SIZE) -> IMAGE_SIZE:
    if isinstance(output_size, int):

        h, w = input_size
        if w > h:
            new_h, new_w = self._output_size, self._output_size * w / h
        else:
            new_h, new_w = self._output_size * h / w, self._output_size
        new_h, new_w = self._output_size
        new_h, new_w = int(new_h), int(new_w)

    return size