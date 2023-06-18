import pandas as pd
data=pd.read_csv('Iris.csv')

data=data.drop('Id', axis=1)

num_data=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for col in num_data:
    # min max normalization
    data[col]=(data[col]-data[col].min())/(data[col].max()-data[col].min())

y=pd.get_dummies(data.pop('Species')).values
x=data.values

import random
import numpy as np

xy=list(zip(x,y))
random.shuffle(xy)
x,y=zip(*xy)
x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

model.add(Dense(1024, activation='relu', input_shape=(4,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history=model.fit(
        x_train, y_train,
        
        validation_data=(x_test, y_test),
        batch_size=32,
        epochs=30)

import matplotlib.pyplot as plt
def show_train_history(train_history):
    plt.figure(figsize=(10,5))
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.xticks([i for i in range(len(train_history.history['acc']))])
    plt.title('Train History')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.xticks([i for i in range(len(train_history.history['loss']))])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(history)

from sklearn.metrics import confusion_matrix
def show_confusion_matrix(cnf_matrix, classes_num):
    cnf_matrix=cnf_matrix.astype('float')/cnf_matrix.sum(axis=1)
    plt.figure(figsize=(5,5))
    plt.imshow(cnf_matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks([i for i in range(classes_num)])
    plt.yticks([i for i in range(classes_num)])
    thresh = cnf_matrix.max() / 2.
    for i in range(classes_num):
        for j in range(classes_num):
            plt.text(
                    j, i,
                    format(cnf_matrix[i, j]*100, '.1f')+'%',
                    horizontalalignment="center",
                    color="white" if cnf_matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
y_true=np.argmax(y_test, axis=1)
y_pred=np.argmax(model.predict(x_test), axis=1)
cnf_matrix = confusion_matrix(y_true, y_pred)
show_confusion_matrix(cnf_matrix, 3)

