import cv2

def show_image(image, win_name=""):

    cv2.imshow(win_name, image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()