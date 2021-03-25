import numpy as np
import cv2
from sudoku.board import Board
from sudoku.puzzle import find_puzzle, extract_digit
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model('model.h5')



image = cv2.imread('sudoku.png')


cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=False)
            
            # verify that the digit is not empty
            if digit is not None:
                # resize the cell to 32x32 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(cell, (32,32))
                cv2.imshow("cell", roi)
                cv2.waitKey(0)
                roi = roi.astype("float32") / 255.0
                roi = img_to_array(roi)
                roi = roi.reshape(1, 32, 32, 1)

                print(roi)
                # roi = np.expand_dims(roi, axis=0)
                # classify the digit and update the Sudoku board with the