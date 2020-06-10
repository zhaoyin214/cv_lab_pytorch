from common.define import ImageSize, Landmarks

def norm_01(image_size: ImageSize, landmarks: Landmarks) -> Landmarks:

    w, h = image_size
    landmarks[: : 2] /= w
    landmarks[1 : : 2] /= h

    return landmarks