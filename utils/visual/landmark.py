import cv2

def show_landmarks(image, landmarks):

    image = image.copy()
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

    cv2.imshow("landmarks", image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
