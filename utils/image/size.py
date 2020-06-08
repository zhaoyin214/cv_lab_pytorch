from common.define import ImageSize

def convert_resize(input_size: ImageSize, output_size: ImageSize) -> ImageSize:

    assert len(input_size) == 2

    if isinstance(output_size, int):
        w, h = input_size
        if w > h:
            new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size * h / w, output_size

        output_size = int(new_w), int(new_h)

    return output_size

def convert_size(size: ImageSize) -> ImageSize:
    if isinstance(size, int):
        size = (size, size)
    else:
        assert len(size) == 2
    return size
