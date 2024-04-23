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

    cv2.imshow("blur", blurred)
    cv2.waitKey()
    canny = cv2.Canny(blurred, 80, 250)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)

    # find the non-zero min-max coords of canny
    pts = np.argwhere(canny > 0)
    if(len(pts) == 0):
        announcer("Image is too Blur, Cannot detect the Currency")
        return

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    # crop the region
    cropped = img[x1:x2, y1:y2]
    # cv2.imwrite("../currency_notes/cropped.png", cropped)
    cv2.imshow("croppred", cropped)
    cv2.waitKey(0)
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
    if (len(pts) == 0):
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
    print("vertices of the given image:")
    vertices = [[y2, sum_x_y2], [sum_y_x2, x2], [y1, sum_x_y1], [sum_y_x1, x1]]

    print(vertices)

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
    #
    # cv2.imshow("Rotated", Rotated_image)
    # cv2.waitKey(0)
    return Rotated_image


#


def currency_detection():
    # begin = time.time()
    max_val = 8
    # max_pt = -1
    # max_kp = 0

    orb = cv2.ORB_create(nfeatures=2500)
    capt = cv2.VideoCapture(0)
    if capt.isOpened():
        ret, frame = capt.read()
    else:
        ret = False

    while ret:
        print("Capturing image...")
        ret, frame = capt.read()
        # img1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("sample511.jpg", frame)
        break
    capt.release()
    cv2.destroyAllWindows()

    test_img = cv2.imread('sample501.jpg')
    cv2.imshow("original", test_img)
    cv2.waitKey(0)
    c_img = cropped_image(test_img)
    if c_img is None:
        announcer("Please Try Again.")
        exit(75)
    cv2.imshow("img1", c_img);
    cv2.waitKey(0)
    r_img = img_rotation(c_img)
    cv2.imshow("img1", r_img);
    cv2.waitKey(0)

    # cv2.imshow("original", test_img)
    # cv2.waitKey(0)
    # cv2.imshow("cropped", c_img)
    # cv2.waitKey(0)

    c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

    (kp1, des1) = orb.detectAndCompute(c_img, None)

    # img_kp1 = cv2.drawKeypoints(test_img,kp1,test_img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite("sample_kp1.jpg",img_kp1)
    # cv2.imshow("keypts",img_kp1)
    # cv2.waitKey(0)

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

        print(i, ' ', f, ' ', len(good))

    if max_val != 8:

        # print('good matches ', max_val)

        # cv2.imshow('test_sample', c_img)
        # cv2.waitKey(0)
        #
        # cv2.imshow('detected_image', train_img)
        # cv2.waitKey(0)
        # print(g, max_val)
        print("maximum number of matches using ORB and image rotation: ", max_val)
        k = ""
        for i in g:
            if ord(i) >= ord('0') and ord(i) <= ord('9'):
                k = k + i
            else:
                break
        print(k)
        if (k[-1] == '1'):
            rup = int(k)
            rup = rup - 1
            announcer(str(rup) + " Rupees is Detected")
        else:
            announcer(k + " Rupees is Detected")

    else:
        print('No Matches')
        # cv2.imshow('sample', test_img)
        # cv2.waitKey(0)
        announcer("Unable to Detect please try again")

    # time.sleep(1)
    # end = time.time()

    # print("time taken by ORB and edge processing:")
    # print(end - begin)


currency_detection()
