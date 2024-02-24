import cv2 
import numpy as np 

pt1 = [450, 300] 
pt2 = [650, 300]
pt3 = [150, 450]
pt4 = [880, 450]

img = cv2.imread("Screenshot 2024-01-21 164150.png")

pts1 = np.float32([pt1, pt2, pt3, pt4])
pts2 = np.float32([[0, 0], [300, 0], [0, 500], [300, 500]])
M = cv2.getPerspectiveTransform(pts1, pts2)
transform_img = cv2.warpPerspective(img, M, (300, 500))
# cv2.circle(img, pt1, 5, (0, 0, 255), -1)
# cv2.circle(img, pt2, 5, (0, 0, 255), -1)
# cv2.circle(img, pt3, 5, (0, 0, 255), -1)
# cv2.circle(img, pt4, 5, (0, 0, 255), -1)

gray = cv2.cvtColor(transform_img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

hsv2 = hsv.copy()
hsv2[:, :, 2] = hsv2[:, :, 2] * 0.4

img_ = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

transform_img = cv2.warpPerspective(img_, M, (300, 500))
ret, thresh = cv2.threshold(transform_img, 30, 255, cv2.THRESH_BINARY)

# cv2.imshow("img", hsv2)
cv2.imshow("hsv", thresh)
cv2.imshow("img_", img_)
# cv2.imshow("thresh", thresh)
# cv2.imshow("transform_img", transform_img)
# cv2.imshow("hsv", hsv)
# cv2.imshow("h", h)
# cv2.imshow("s", s)
# cv2.imshow("v", v)
cv2.waitKey(0)