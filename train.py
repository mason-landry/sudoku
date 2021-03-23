import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
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

    ## Load train folder as training set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data,
        image_size=(32, 32),
        batch_size=16,
        color_mode='grayscale',
        shuffle=True,
        validation_split=0.2,
        seed=123,
        subset='training')

    ## Load train folder as training set
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data,
        image_size=(32, 32),
        batch_size=16,
        color_mode='grayscale',
        shuffle=True,
        validation_split=0.2,
        seed=123,
        subset='validation')
    
    ## Initialize model
    model = SudokuNet.build(width=32, height=32, depth=1, classes=9)    
    # compile model
    opt = SGD(lr=0.00001, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    H = model.fit(train_ds, epochs=75, verbose=1, validation_data=test_ds)
    # Show curves
    summarize_diagnostics(H)
    # serialize the model to disk
    print("[INFO] serializing digit model...")
    model.save("model.h5")


