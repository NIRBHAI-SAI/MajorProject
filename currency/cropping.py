import cv2
import numpy as np

def cropped_image(img):

    #function that generates the cropped images
    blurred = cv2.blur(img, (5, 5))  # bluring the image

    cv2.imshow("blur", blurred)
    cv2.waitKey()
    canny = cv2.Canny(blurred, 80, 250)
    cv2.imshow("canny",canny)
    cv2.waitKey(0)

    # find the non-zero min-max coords of canny
    pts = np.argwhere(canny > 0)

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    # crop the region
    cropped = img[x1:x2, y1:y2]
    # cv2.imwrite("../currency_notes/cropped.png", cropped)
    cv2.imshow("croppred",cropped)
    cv2.waitKey(0)
    return cropped

# img = cv2.imread("sample.jpg")
# cropped_image(img)

