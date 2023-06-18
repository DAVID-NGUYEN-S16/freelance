import numpy as np 

#以下兩種載入方式二擇一，依照OpenCV安裝過程不同，可能需要不同的載入方式
from cv2 import cv2
# import cv2
 
imageName = "lena.jpg"
img = cv2.imread(imageName, cv2.IMREAD_COLOR)

# 設定kernel size為3x3
kernel_size = 3

# Sobel
kernel_H = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
kernel_V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

print(kernel_H)
print(kernel_V)

# 使用cv2.filter2D進行convolute，
img_H = cv2.filter2D(img, ddepth=3, dst=-1, kernel=kernel_H, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
img_V = cv2.filter2D(img, ddepth=3, dst=-1, kernel=kernel_V, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
cv2.imwrite("convoluted_H.jpg", img_H)
cv2.imwrite("convoluted_V.jpg", img_V)