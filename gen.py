import numpy as np
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from pathlib import Path
import cv2

def generate(num = 5000, pth=".\\data\\"):

    
    for i in range(1,10):
        ## Make sure directory contains folders 1-9 and train/test folders
        Path(pth + str(i)).mkdir(parents=True, exist_ok=True)

        # generate <num> images and save them in train folder
        print('creating images for', i)
        for j in range(num):
            fnt = np.random.choice(glob(".\\sudoku\\fonts\\*"))
            fnt = ImageFont.truetype(font=fnt, size = 160)
            img = Image.new('L', (200, 200), color='white')
            d = ImageDraw.Draw(img)
            d.text((55,20),str(i), font=fnt)

            img.thumbnail((32,32))
            path = pth + str(i) + '\\'
            img.save(path + str(j) + '.png')

            img = cv2.imread(path + str(j) + '.png')

            
            # apply automatic thresholding to the cell and then clear any
            # connected borders that touch the border of the cell
            thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]

            # # find contours in the thresholded cell
            # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = imutils.grab_contours(cnts)
            # # if no contours were found than this is an empty cell
            # if len(cnts) == 0:
            #     return None

            # # otherwise, find the largest contour in the cell and create a
            # # mask for the contour
            # c = max(cnts, key=cv2.contourArea)
            # mask = np.zeros(thresh.shape, dtype="uint8")
            # cv2.drawContours(mask, [c], -1, 255, -1)

            # # compute the percentage of masked pixels relative to the total
            # # area of the image
            # (h, w) = thresh.shape
            # percentFilled = cv2.countNonZero(mask) / float(w * h)

            # # if less than 3% of the mask is filled then we are looking at
            # # noise and can safely ignore the contour
            # if percentFilled < 0.03:
            #     return None
            
            # apply the mask to the thresholded cell

            img = cv2.bitwise_and(thresh, thresh)
            cv2.imwrite(path + str(j) + '.png', img)
            
        

if __name__ == "__main__":
    generate()
    


