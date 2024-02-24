import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

pt1 = [450, 300] 
pt2 = [650, 300]
pt3 = [150, 450]
pt4 = [880, 450]

img = cv2.imread("Screenshot 2024-01-21 164150.png")

pts1 = np.float32([pt1, pt2, pt3, pt4])
pts2 = np.float32([[0, 0], [300, 0], [0, 500], [300, 500]])
M = cv2.getPerspectiveTransform(pts1, pts2)
transform_img = cv2.warpPerspective(img, M, (300, 500))

gray_transform_img = cv2.cvtColor(transform_img, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray_transform_img)
equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)

gamma = 0.1
gamma_corrected_img = np.power(transform_img/255.0, gamma)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray_transform_img)

min_val = clahe_img.min()
max_val = clahe_img.max()
compressed_img = ((clahe_img - min_val) / (max_val - min_val))

gamma = 0.8
gamma_corrected_img = np.power(clahe_img/255.0, gamma)
gamma_corrected_img = (gamma_corrected_img * 255).astype(np.uint8)

print(len(gamma_corrected_img.shape))
# gamma_corrected_img = np.reshape(gamma_corrected_img, (gamma_corrected_img.shape[0], gamma_corrected_img.shape[1], 1))
if len(gamma_corrected_img.shape) == 2:   # if grayscale image, convert.
    print("gamma length :", len(gamma_corrected_img))
    rgb = cv2.cvtColor(gamma_corrected_img, cv2.COLOR_GRAY2RGB)
    print(rgb.shape)
retval, bi = cv2.threshold(gamma_corrected_img, 220, 255, cv2.THRESH_BINARY)




# hist_b = cv2.calcHist(transform_img, [0], None, [256], [0, 256])
# hist_g = cv2.calcHist(transform_img, [1], None, [256], [0, 256])
# hist_r = cv2.calcHist(transform_img, [2], None, [256], [0, 256])

# plt.clf()
# plt.plot(hist_b, "b")
# plt.plot(hist_g, "g")
# plt.plot(hist_r, "r")
# plt.pause(1)
# cv2.imshow('equal', rgb)
cv2.imshow("binary", bi)
# print(bi.shape)
cv2.waitKey(0)