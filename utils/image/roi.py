from common.define import Box, Image

def crop(image: Image, box: Box) -> Image:
    return image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]