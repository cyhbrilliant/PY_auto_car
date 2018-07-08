import numpy as np
import cv2
def findline(image):
    # image = cv2.resize(image, dsize=(320, 240))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # gray = cv2.equalizeHist(gray)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 150
    high_threshold = 400
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges[0:120, :] = 0
    edges[:, :60] = 0
    edges[:, 300:] = 0
    edges[220:240, :] = 0

    # mask = np.zeros_like(edges)
    # ignore_mask_color = 255
    # # This time we are defining a four sided polygon to mask
    # imshape = image.shape
    # vertices = np.array([[(50,imshape[0]),(420, 280), (550, 280), (950,imshape[0])]], dtype=np.int32)
    # cv2.fillPoly(mask, vertices, ignore_mask_color)
    # masked_edges = cv2.bitwise_and(edges, mask)

    rho = 5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    def linefilter(pt1, pt2):
        if pt1[0] > pt2[0]:
            return False
        if abs(pt1[1] - pt2[1]) < 20:
            return False

        return True

    imgcopy = image.copy()
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    # if type(lines) != type(None):
    if lines is not None:
        for line in lines:
            pt1 = (line[0][0], line[0][1])
            pt2 = (line[0][2], line[0][3])
            if linefilter(pt1, pt2):
                cv2.line(imgcopy, pt1, pt2, (0, 0, 255), 5)

    # print(lines)

    # cv2.imshow('1', blur_gray)
    # cv2.imshow('2', edges)
    # cv2.imshow('3', image)
    # cv2.waitKey(0)
    return imgcopy