import cv2

from .image import show_image

def show_landmarks(image, landmarks):

    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = image.shape[0 : 2]

    for idx in range(len(landmarks) // 2):

        x = int(landmarks[2 * idx] * w)
        y = int(landmarks[2 * idx + 1] * h)
        cv2.circle(
            img=image,
            center=(int(x), int(y)),
            radius=1,
            color=(0, 255, 0),
            thickness=-1,
            lineType=cv2.FILLED
        )

    show_image(image, "landmarks")
