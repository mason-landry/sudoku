import numpy as np
import cv2
from sudoku.board import Board
from sudoku.puzzle import find_puzzle, extract_digit
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    # load the digit classifier from disk
    print("[INFO] loading digit classifier...")
    model = load_model('model.h5')

    image = cv2.imread('sudoku1.jpg')
    puzzle, warped = find_puzzle(image, debug=False)
    # Initialize empty board
    board = Board()

    # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []

    # loop over the grid locations
    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))

            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell, debug=False)
            
            # verify that the digit is not empty
            if digit is not None:
                # resize the cell to 32x32 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(digit, (32,32))
                cv2.imshow("digit", roi)
                cv2.waitKey(0)
                roi = roi.astype("float32") / 255.0
                roi = img_to_array(roi)
                roi = roi.reshape(1, 32, 32, 1)

                # roi = np.expand_dims(roi, axis=0)
                # classify the digit and update the Sudoku board with the
                
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]
                print(pred)
                board.update(pred, y, x)

    board.show()
    board.solve()
    print("- - - - - - - - - ")
    board.show()

