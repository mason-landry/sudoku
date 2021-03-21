import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from model.model import SudokuNet


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



if __name__ == "__main__":
    ## Define train and test paths
    train = '.\\data\\train'
    test = '.\\data\\test'

    ## Load train folder as training set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train,
        image_size=(28, 28),
        batch_size=16)
    # tf.reshape(train_ds, [60, 60])

    ## Load test folder as testing dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test,
        image_size=(28, 28),
        batch_size=16)
    # tf.reshape(test_ds, [60, 60])
    
    ## Initialize model
    model = SudokuNet.build(width=28, height=28, depth=3, classes=9)    
    opt = Adam(lr=0.001)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], )

    H = model.fit(train_ds, epochs=10, verbose=1)

    # serialize the model to disk
    print("[INFO] serializing digit model...")
    model.save("model.h5")


