import pandas as pd
data=pd.read_csv('boston_housing.csv')

unuse=['zn', 'nox', 'rm', 'age', 'ptratio', 'black']
data=data.drop(unuse, axis=1)

y=data.pop('medv').values.astype('float32')
x=data.values.astype('float32')

import random
import numpy as np
xy=list(zip(x,y))
random.shuffle(xy)
x,y=zip(*xy)
x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_mean=np.mean(x_train, axis=0)
x_std=np.std(x_train, axis=0)
y_mean=np.mean(y_train, axis=0)
y_std=np.std(y_train, axis=0)

x_train=(x_train-x_mean)/x_std
x_test=(x_test-x_mean)/x_std
y_train=(y_train-y_mean)/y_std
y_test=(y_test-y_mean)/y_std

x_min=np.min(x_train, axis=0)
x_max=np.max(x_train, axis=0)
y_min=np.min(y_train, axis=0)
y_max=np.max(y_train, axis=0)
x_train=(x_train-x_min)/(x_max-x_min)
x_test=(x_test-x_min)/(x_max-x_min)
y_train=(y_train-y_min)/(y_max-y_min)
y_test=(y_test-y_min)/(y_max-y_min)

x_train=x_train.reshape(x_train.shape+(1,))
x_test=x_test.reshape(x_test.shape+(1,))
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.models import Sequential

model=Sequential()
model.add(Conv1D(
        32, 
        kernel_size=2, 
        padding='same', 
        activation='relu', 
        input_shape=(7, 1)))
model.add(Dropout(rate=0.3))
model.add(Conv1D(
        32, 
        kernel_size=2, 
        padding='same', 
        activation='relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.1))
model.add(Dense(1, activation='linear'))

import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

model.compile(
        loss=rmse,
        optimizer='rmsprop')
history=model.fit(
        x_train,y_train,
        validation_data=(x_test,y_test),
        epochs=60)

import matplotlib.pyplot as plt
def show_train_history(train_history):
    plt.figure(figsize=(10,5))
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.xticks([i for i in range(len(train_history.history['loss']))])
    plt.title('Train History')
    plt.ylabel('loss(rmse)')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(history)

def inverse(y):
    inverse_y=y*(y_max-y_min)+y_min
    inverse_y=inverse_y*y_std+y_mean
    inverse_y[np.where(inverse_y<0)]=0
    return inverse_y
y_true=inverse(y_test)
y_pred=model.predict(x_test).ravel()
y_pred=inverse(y_pred)

plt.figure(figsize=(20,5))
plt.bar([i for i in range(len(y_true))], y_true, alpha=0.5, color='g')
plt.bar([i for i in range(len(y_pred))], y_pred, alpha=0.5, color='y')
plt.xticks([i for i in range(len(y_true))])
plt.legend(['lower than true', 'higher than true'], loc='upper left')
plt.show()

from sklearn.metrics import mean_squared_error
print('rmse: ', np.sqrt(mean_squared_error(y_true,y_pred)))
