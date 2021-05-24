import os
import numpy as np

from pathlib import Path
from keras import models
from keras.utils import to_categorical
from keras.layers import Dense,Flatten
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.applications.vgg16 import VGG16

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

def buildModel(typeFlowers, x_train, y_train, batch_size=8, epochs=8):
    FILE_MODLE = "model_VGG16"

    vgg16=VGG16(
        input_shape=[64,64,3],
        weights='imagenet',
        include_top=False
    )
    
    # freeze the layers in base model
    for layer in vgg16.layers:
        layer.trainable = False
    
    x = Flatten()(vgg16.output)
    x = Dense(len(typeFlowers), activation='softmax')(x)
    
    model = models.Model(inputs=vgg16.input, outputs=x)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_split=0.1)
    model.save(FILE_MODLE)
    
    return model

def train():
    FILE_MODLE = "model_VGG16"
    if os.path.isfile(FILE_MODLE):
        model = models.load_model(FILE_MODLE)
    else:
        dataPath = "./flowers"
        labels, images_data, typeFlowers = readData(dataPath=dataPath)
        
        x_train, x_test, y_train, y_test = train_test_split(images_data, labels, test_size=0.25)
        x_train = np.reshape(x_train, [-1, 64, 64, 3])
        x_train = x_train/255.0
        x_test = np.reshape(x_test, [-1, 64, 64, 3])
        x_test = x_test/255.0
        y_train = to_categorical(y_train)

        model = buildModel(typeFlowers=typeFlowers, x_train=x_train, y_train=y_train, batch_size=128, epochs=30)
        print("answer test")
        y_pred = np.argmax(model.predict(x_test), axis=1)

        print(classification_report(y_test, y_pred, target_names=typeFlowers))
    return model

if __name__ == '__main__':
    train()
    model = train()
    for idx, type_flower in enumerate(os.listdir("./flowers")):
        print(idx, type_flower)
    
    path = "./flowers/dandelion/33876197394_c7a9487a9f_n.jpg"
    img = image.img_to_array(image.load_img(path, target_size = (64, 64)))
    x_test = np.reshape(img, [-1, 64, 64, 3])
    x_test = x_test/255.0
    
    print(np.argmax(model.predict(x_test), axis=1))