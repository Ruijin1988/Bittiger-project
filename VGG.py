import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
import matplotlib.pylab as plt
import numpy as np
# import theano.tensor.nnet.abstract_conv as absconv
import cv2
import h5py
import os

def VGG(mask_size, num_input_channels=1024):
    """
    Build Convolution Neural Network
    args : nb_classes (int) number of classes
    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 360, 640)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # FC
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(mask_size, activation='linear'))

    model.name = "VGG"

    return model

def vgg_loss(Y, Y_p):
    loss=Y-Y_p;
    ssq = 4*np.sum(loss**2)
    return ssq

def vgg_error(y_true, y_pred):
    return 4*K.sum(K.square(y_pred - y_true))

import scipy.io as sio
mask=sio.loadmat('mask_5000.mat')['mask']
image=sio.loadmat('image_5000.mat')['image_store']

Train_data=image.reshape(5000,1,360,640)
X_train=Train_data[0:1];X_test=Train_data[1:2]
Y_train=mask[0:1];Y_test=mask[1:2]

model=VGG(mask_size=36*64)
model.compile(optimizer='sgd',loss='mse')
model.fit(X_train,Y_train,shuffle=True,nb_epoch=10,batch_size=1,validation_data=(X_test,Y_test))