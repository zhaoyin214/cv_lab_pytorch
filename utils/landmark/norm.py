from common.define import ImageSize, Landmarks

def norm_01(ImageSize: ImageSize, landmarks: Landmarks) -> Landmarks:

    h, w = ImageSize
    for idx in range(len(landmarks) // 2):
        landmarks[2 * idx] /= w
        landmarks[2 * idx + 1] /= h

    return landmarks