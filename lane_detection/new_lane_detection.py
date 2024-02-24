import cv2  
import numpy as np 
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture("test_video.mp4")
cap = cv2.VideoCapture("challenge.mp4")

# pt1 = [440, 400] 
# pt2 = [750, 400]
# pt3 = [290, 600]
# pt4 = [950, 600] 

pt1 = [600, 450] 
pt2 = [750, 450]
pt3 = [200, 650]
pt4 = [1100, 650]


while True:
    ret, frame = cap.read()
    # print(frame.shape)
    # cv2.circle(frame, pt1, 5, (0, 0, 255), -1)
    # cv2.circle(frame, pt2, 5, (0, 0, 255), -1)
    # cv2.circle(frame, pt3, 5, (0, 0, 255), -1)
    # cv2.circle(frame, pt4, 5, (0, 0, 255), -1)
    pts1 = np.float32([pt1, pt2, pt3, pt4])
    pts2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transform_frame = cv2.warpPerspective(frame, M, [400, 640])
    
    # gray = cv2.cvtColor(transform_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(transform_frame, 100, 255, cv2.THRESH_BINARY)
    kernels = np.ones((7, 7), np.uint8)
    thresh2 = cv2.erode(thresh, kernels, 2)
    kernels = np.ones((100, 7), np.uint8)
    thresh2 = cv2.dilate(thresh2, kernels, 2)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(thresh, 200, 255)

    # Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    y = 700
    lx = []
    rx = []

    mask2 = mask.copy()

    while y > 0:
        # Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50+cx)
                left_base = left_base-50 + cx

        # Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                rx.append(right_base-50+cx)
                right_base = right_base-50 + cx

        cv2.rectangle(transform_frame, (left_base-5, y), (left_base+5, y-40), (0, 0, 255), 10)
        cv2.rectangle(transform_frame, (right_base-5, y), (right_base+5, y-40), (0, 0, 255), 10)
        y -= 40
    # histogram = np.sum(canny, axis=0)
    # plt.clf()
    # plt.plot(histogram)
    # plt.pause(0.01)
    
    M2= cv2.getPerspectiveTransform(pts2, pts1)
    transform_frame2 = cv2.warpPerspective(transform_frame, M2, [frame.shape[1], frame.shape[0]])
    merge = cv2.bitwise_or(frame, transform_frame2)

    cv2.imshow("frame", merge)
    cv2.imshow('perspective', transform_frame)
    cv2.imshow('thresh', thresh)
    cv2.imshow('erod', thresh2)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break