from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img = x_train[0]
plt.gcf().set_size_inches(17, 5)
plt.subplot(121) 
plt.imshow(img)
plt.subplot(122) 
plt.hist(img[:,:,0].ravel(), 256, color='b') 
plt.hist(img[:,:,1].ravel(), 256, color='g')
plt.hist(img[:,:,2].ravel(), 256, color='r')
plt.show()
