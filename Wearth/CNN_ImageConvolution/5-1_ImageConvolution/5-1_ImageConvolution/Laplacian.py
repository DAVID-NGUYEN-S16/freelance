import numpy as np

from cv2 import cv2
# import cv2
 
imageName = "lena.jpg"
img = cv2.imread(imageName, cv2.IMREAD_COLOR)

kernel_size = 3

kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

print (kernel1)
print (kernel2)

# use cv2.filter2 convolute，
img1 = cv2.filter2D(img, ddepth=3, dst=-1, kernel=kernel1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
img2 = cv2.filter2D(img, ddepth=3, dst=-1, kernel=kernel2, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
cv2.imwrite("convoluted1.jpg", img1)
cv2.imwrite("convoluted2.jpg", img2)