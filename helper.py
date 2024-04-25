import os
import cv2
from tts.announce import announcer
import time
import numpy as np
import math
import imutils


def cropped_image(img):
    # function that generates the cropped images
    blurred = cv2.blur(img, (5, 5))  # bluring the image

    # cv2.imshow("blur", blurred)
    # cv2.waitKey()
    canny = cv2.Canny(blurred, 80, 250)
    # cv2.imshow("canny", canny)
    # cv2.waitKey(0)

    # find the non-zero min-max coords of canny
    pts = np.argwhere(canny > 0)
    if len(pts) == 0:
        announcer("Image is too Blur, Cannot detect the Currency")
        return

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    # crop the region
    cropped = img[x1:x2, y1:y2]
    # cv2.imwrite("../currency_notes/cropped.png", cropped)
    # cv2.imshow("croppred", cropped)
    # cv2.waitKey(0)
    return cropped


def img_rotation(img):
    blurred = cv2.blur(img, (5, 5))  # bluring the image

    # cv2.imshow("blur", blurred)
    # cv2.waitKey()
    canny = cv2.Canny(blurred, 80, 250)
    # cv2.imshow("canny", canny)
    # cv2.waitKey()
    # find the non-zero min-max coords of canny
    pts = np.argwhere(canny > 0)
    if len(pts) == 0:
        return

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    # print(y1)
    # print(x1)
    # print(y2)
    # print(x2)

    req_y2 = np.where(pts == y2)
    # print(req_y2)
    n = len(req_y2[0])
    sum_x_y2 = 0
    count = 0
    for i in range(n):
        if req_y2[1][i] == 1:
            idx = req_y2[0][i]
            sum_x_y2 = sum_x_y2 + pts[idx][0]
            count = count + 1
    sum_x_y2 = sum_x_y2 // count
    # print("vertex 1 :" + "(" + str(y2) + "," + str(sum_x_y2) + ")")

    req_y1 = np.where(pts == y1)
    # print(req_y1)
    n = len(req_y1[0])
    sum_x_y1 = 0
    count = 0
    for i in range(n):
        if req_y1[1][i] == 1:
            idx = req_y1[0][i]
            sum_x_y1 = sum_x_y1 + pts[idx][0]
            count = count + 1
    sum_x_y1 = sum_x_y1 // count
    # print("vertex 2 :" + "(" + str(y1) + "," + str(sum_x_y1) + ")")

    req_x1 = np.where(pts == x1)
    n = len(req_x1[0])
    sum_y_x1 = 0
    count = 0
    for i in range(n):
        if req_x1[1][i] == 0:
            idx = req_x1[0][i]
            sum_y_x1 = sum_y_x1 + pts[idx][1]
            # print(pts[idx][0])
            count = count + 1
    sum_y_x1 = sum_y_x1 // count
    # print("vertex 3 :" + "(" + str(sum_y_x1) + "," + str(x1) + ")")

    req_x2 = np.where(pts == x2)
    n = len(req_x2[0])
    sum_y_x2 = 0
    count = 0
    for i in range(n):
        if req_x2[1][i] == 0:
            idx = req_x2[0][i]
            sum_y_x2 = sum_y_x2 + pts[idx][1]
            # print(pts[idx][0])
            count = count + 1
    sum_y_x2 = sum_y_x2 // count
    # print("vertex 4 :" + "(" + str(sum_y_x2) + "," + str(x2) + ")")
    # print("vertices of the given image:")
    vertices = [[y2, sum_x_y2], [sum_y_x2, x2], [y1, sum_x_y1], [sum_y_x1, x1]]

    # print(vertices)

    mid_y1 = (vertices[0][1] + vertices[1][1]) // 2
    mid_x1 = (vertices[0][0] + vertices[1][0]) // 2
    mid_y2 = (vertices[2][1] + vertices[3][1]) // 2
    mid_x2 = (vertices[2][0] + vertices[3][0]) // 2
    # print(mid_x1, mid_y1)
    # print(mid_x2, mid_y2)
    slope = (mid_y2 - mid_y1) / (mid_x2 - mid_x1)
    deg = math.atan(slope) * (180 / math.pi)
    img = img[x1:x2, y1:y2]

    Rotated_image = imutils.rotate(img, angle=deg)
    # cv2.imwrite("rotated.jpg",Rotated_image)
    # cv2.imshow("Rotated", Rotated_image)
    # cv2.waitKey(0)
    return Rotated_image


#


def currency_detection(tst):
    max_val = 8

    orb = cv2.ORB_create(nfeatures=2500)

    test_img = cv2.imread(tst)

    c_img = cropped_image(test_img)
    if c_img is None:
        announcer("Please Try Again.")
        exit(75)

    # r_img = img_rotation(c_img)

    c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

    (kp1, des1) = orb.detectAndCompute(c_img, None)

    path = 'notes'
    i = 0
    g = os.listdir(path)[0]
    for f in sorted(os.listdir(path)):
        # train image
        i = i + 1
        train_img = cv2.imread(os.path.join(path, f))
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

        (kp2, des2) = orb.detectAndCompute(train_img, None)
        # brute force matcher
        bf = cv2.BFMatcher()
        all_matches = bf.knnMatch(des1, des2, k=2)

        good = []

        for (m, n) in all_matches:
            if m.distance < 0.789 * n.distance:
                good.append([m])

        if len(good) > max_val:
            max_val = len(good)
            max_pt = i
            g = f
            max_kp = kp2

        # print(i, ' ', f, ' ', len(good))

    if max_val != 8:

        # print("maximum number of matches using ORB and image rotation: ", max_val)
        k = ""
        for i in g:
            if i == ' ':
                break
            if ord('0') <= ord(i) <= ord('9'):
                k = k + i
            else:
                break
        # print(k)
        return k

    else:
        print('No Matches')
        # cv2.imshow('sample', test_img)
        # cv2.waitKey(0)
        # announcer("Unable to Detect please try again")
        return "0"