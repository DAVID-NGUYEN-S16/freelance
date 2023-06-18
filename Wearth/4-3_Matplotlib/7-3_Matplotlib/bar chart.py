from tensorflow.keras.datasets import cifar10
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test)=cifar10.load_data()
img=cv2.cvtColor(x_train[0], cv2.COLOR_BGR2GRAY)
img=img.ravel()
x=[i for i in range(255)]
y=[len(np.where(img==i)[0]) for i in range(255)] 
plt.gcf().set_size_inches(18.5, 5) 
plt.bar(x, y)
plt.ylim((min(y), max(y))) 
plt.show()
