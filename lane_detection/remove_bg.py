import cv2 
import numpy as np


pt1 = [850, 650] 
pt2 = [1130, 650]
pt3 = [500, 850]
pt4 = [1500, 850]

img = cv2.imread("Screenshot 2024-01-21 154729.png")
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 500, 300)

# cv2.circle(img, pt1, 5, (0, 0, 255), -1)
# cv2.circle(img, pt2, 5, (0, 0, 255), -1)
# cv2.circle(img, pt3, 5, (0, 0, 255), -1)
# cv2.circle(img, pt4, 5, (0, 0, 255), -1)

pts1 = np.float32([pt1, pt2, pt3, pt4])
pts2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])
M = cv2.getPerspectiveTransform(pts1, pts2)
transform_frame = cv2.warpPerspective(img, M, [400, 640])

ret, threshold = cv2.threshold(transform_frame, 100, 255, cv2.THRESH_TOZERO)
cv2.imshow("Image", threshold)
cv2.waitKey(0)
