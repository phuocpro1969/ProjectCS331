import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from keras import models
from keras.utils import to_categorical
from keras.layers import Conv2D,Dense,Dropout,Input,Flatten,MaxPooling2D
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def readData(dataPath):
    typeFlowers = os.listdir(dataPath)
    images_data = []
    labels = []
    
    for idx, typeFlower in enumerate(typeFlowers):
        link =Path(dataPath, typeFlower)
        for path in link.glob("*.jpg"):
            img = image.img_to_array(image.load_img(path, target_size = (64, 64)))
            images_data.append(img)
            labels.append(idx)
    return np.array(labels), np.array(images_data), typeFlowers

def buildModel(x_train, y_train, batch_size=8, epochs=8):
    if os.path.isfile("model_CNN"):
        model = models.load_model("model_CNN")
    else:
        input_shape = (64, 64, 3)
        inputs = Input(shape = input_shape)
        x = inputs
        x = Conv2D(32, kernel_size=2, activation='relu', strides=1, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, kernel_size=2, activation='relu', strides=1, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)


        x = Conv2D(128, kernel_size=2, activation='relu', strides=1, padding='same')(x)
        x = Conv2D(256, kernel_size=2, activation='relu', strides=1, padding='same')(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        outputs = Dense(5,activation='softmax')(x)
        model = models.Model(inputs, outputs)

        model.summary()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
        model.save("model_CNN")
    return model

def main():
    dataPath = "./flowers"
    labels, images_data, typeFlowers = readData(dataPath=dataPath)
    
    x_train, x_test, y_train, y_test = train_test_split(images_data, labels, test_size=0.25)
    x_train = np.reshape(x_train, [-1, 64, 64, 3])
    x_test = np.reshape(x_test, [-1, 64, 64, 3])
    x_train = x_train/255.0
    x_test = x_test/255.0
    y_train = to_categorical(y_train)

    model = buildModel(x_train=x_train, y_train=y_train, batch_size=128, epochs=30)
    y_pred = np.argmax(model.predict(x_test), axis=1)

    print(classification_report(y_test, y_pred, target_names=typeFlowers))

if __name__ == '__main__':
    main()