from common.define import ImageSize, Landmarks

def norm_01(image_size: ImageSize, landmarks: Landmarks) -> Landmarks:

    w, h = image_size
    for idx in range(len(landmarks) // 2):
        landmarks[2 * idx] /= w
        landmarks[2 * idx + 1] /= h

    return landmarks