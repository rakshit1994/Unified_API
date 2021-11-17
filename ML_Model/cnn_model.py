import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import os
from os.path import join, dirname
import json
import gzip
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import Google_PubSub.pubsub as pubsub


def load_data(folder_name,dataset_name,files_list):
    """ Loads the Fashion MNIST dataset
    return: x_train, y_train, x_test, y_test
    """
    dir_name = os.path.join(dirname(__file__),'datasets', 'fashion_mnist')
    print(dir_name)
    files = files_list

    paths = []

    for fname in files:
        paths.append(os.path.join(dir_name, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)


    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    # y_train = [vectorize_data(i) for i in y_train]
    # y_test = [vectorize_data(i) for i in y_test]
    
    return x_train, y_train, x_test, y_test

def train():
    #add/replace in ML_Model/datasets folder and update the line below with your files
    X_train, y_train, X_test, y_test = load_data('datasets', 'fashion_mnist',['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'])
    image_rows = 28
    image_cols = 28
    image_shape = (image_rows,image_cols,1)
    X_train = X_train.reshape(X_train.shape[0],*image_shape)
    X_test = X_test.reshape(X_test.shape[0],*image_shape)

    print("x_train shape = {}".format(X_train.shape))
    print("x_test shape = {}".format(X_test.shape))

    cnn_model = Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
    tf.keras.layers.MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(), # flatten out the layers
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(10,activation = 'softmax')
    
    ])

    cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics =['accuracy'])

    # Train_dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(buffer_size=len(X_train))\
            .batch(batch_size=128)\
            .prefetch(buffer_size=128)\
            .repeat()

    # Test dataset
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))\
                .batch(batch_size=128)\
                .prefetch(buffer_size=128)\
                .repeat()

    cnn_model.fit(train_ds, steps_per_epoch=len(X_train)/128, epochs=2)

    loss, acc = cnn_model.evaluate(test_ds, steps=len(X_test)/128)
    print('test loss is {}'.format(loss))
    print('test accuracy is {}'.format(acc))

    # Save the model
    cnn_model.save(join(dirname(__file__),'saved_model.h5'))

def vectorize_data(y):
    e = np.zeros((10,1))
    e[y] = 1.0
    return e

def predict():
    # Load the model
    model = load_model(join(dirname(__file__), 'saved_model.h5'))
    img_path = join(dirname(__file__),"test_image.jpg")
    img = image.load_img(img_path,color_mode="grayscale",target_size=(28, 28))
    #plt.imshow(img)
    #plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict_classes(x)
    prob = model.predict_proba(x)
    x = np.resize(x, (28,28,1))
    labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
            5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}
    obj = {"prediction":labels_map[prediction[0]],"confidence":prob[0][0]}
    print(obj)


def predict_img(file: Image.Image):
    model = load_model(join(dirname(__file__), 'saved_model.h5'))
    x = np.resize(file, (28,28,1))
    print(np.shape(x))
    x =  np.expand_dims(x, axis=0)
    prediction = model.predict_classes(x)
    prob = model.predict_proba(x)
    labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
            5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}
    obj = {"prediction":labels_map[prediction[0]],"confidence_percentage":round(prob[0][0]*100,4)}
    return pubsub.push_pubsub(json.dumps(obj))
    

#if __name__ == "__main__": #uncomment to test code
    #train() #Uncomment to train again, tweak parameters for cnn to enhance the model
    #predict() #Testing prediction using local file


