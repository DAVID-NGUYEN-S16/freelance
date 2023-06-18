from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

import numpy as np
y_true=np.argmax(y_test, axis=1)
y_pred=np.argmax(model.predict(x_test), axis=1)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_true, y_pred)

cnf_matrix=cnf_matrix.astype('float')/cnf_matrix.sum(axis=1)
print(cnf_matrix.shape)

plt.figure(figsize=(10,10))
plt.imshow(cnf_matrix, cmap='Blues')
plt.colorbar()
plt.xticks([i for i in range(10)])
plt.yticks([i for i in range(10)])

thresh = cnf_matrix.max() / 2.
for i in range(10):
    for j in range(10):
        plt.text(
                j, i,
                format(cnf_matrix[i, j], '.2f')+'%',
                horizontalalignment="center",
                color="white" if cnf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout() 
plt.show()
