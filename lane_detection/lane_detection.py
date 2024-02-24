import cv2  
import numpy as np 
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture("test_video.mp4")
cap = cv2.VideoCapture("challenge.mp4")

# pt1 = [440, 400] 
# pt2 = [750, 400]
# pt3 = [290, 600]
# pt4 = [950, 600]

pt1 = [400, 550] 
pt2 = [950, 550]
pt3 = [250, 650]
pt4 = [1200, 650]

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 0, 255, nothing)

while True:
    ret, frame = cap.read()
    # print(frame.shape)
    cv2.circle(frame, pt1, 5, (0, 0, 255), -1)
    cv2.circle(frame, pt2, 5, (0, 0, 255), -1)
    cv2.circle(frame, pt3, 5, (0, 0, 255), -1)
    cv2.circle(frame, pt4, 5, (0, 0, 255), -1)
    pts1 = np.float32([pt1, pt2, pt3, pt4])
    pts2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transform_frame = cv2.warpPerspective(frame, M, [400, 640])
    # gray = cv2.cvtColor(transform_frame, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # canny = cv2.Canny(thresh, 5, 5)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    hsv_transformed_frame = cv2.cvtColor(transform_frame, cv2.COLOR_BGR2HSV)
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    y = 472
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

        cv2.rectangle(transform_frame, (left_base-20, y), (left_base+20, y-40), (0, 0, 255), 2)
        cv2.rectangle(transform_frame, (right_base-20, y), (right_base+20, y-40), (0, 0, 255), 2)
        y -= 40
    # histogram = np.sum(canny, axis=0)
    # plt.clf()
    # plt.plot(histogram)
    # plt.pause(0.01)

    cv2.imshow("frame", frame)
    cv2.imshow('perspective', transform_frame)
    cv2.imshow('canny', mask2)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break