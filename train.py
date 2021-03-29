import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
import sys

from model.model import SudokuNet


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

if __name__ == "__main__":
    ## Define train and test paths
    data = '.\\data\\'

    # ## Load train folder as training set
    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data,
    #     image_size=(32, 32),
    #     batch_size=16,
    #     color_mode='grayscale',
    #     shuffle=True,
    #     validation_split=0.2,
    #     seed=123,
    #     subset='training')

    # print(train_ds)

    # ## Load train folder as training set
    # test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data,
    #     image_size=(32, 32),
    #     batch_size=16,
    #     color_mode='grayscale',
    #     shuffle=True,
    #     validation_split=0.2,
    #     seed=123,
    #     subset='validation')

    (train_ds, train_labels), (test_ds, test_labels) = mnist.load_data()
    train_ds = train_ds.reshape((train_ds.shape[0], 28, 28, 1))
    test_ds = test_ds.reshape((test_ds.shape[0], 28, 28, 1))
    # one hot encode target values
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    # convert from integers to floats
    train_norm = train_ds.astype('float32')
    test_norm = test_ds.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images

    ## Initialize model
    model = SudokuNet.build(width=28, height=28, depth=1, classes=10)    
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    H = model.fit(train_norm, train_labels, epochs=20, batch_size=64, verbose=1, validation_data=(test_norm, test_labels))
    # Show curves
    summarize_diagnostics(H)
    # serialize the model to disk
    print("[INFO] serializing digit model...")
    model.save("model.h5")


