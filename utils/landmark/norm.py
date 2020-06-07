from common.define import IMAGE_SIZE, LANDMARKS

def norm_01(image_size: IMAGE_SIZE, landmarks: LANDMARKS) -> LANDMARKS:

    h, w = image_size
    for idx in range(len(landmarks) // 2):
        landmarks[2 * idx] /= w
        landmarks[2 * idx + 1] /= h

    return landmarks