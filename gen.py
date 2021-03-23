import numpy as np
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from pathlib import Path
 

def generate(num = 1000, pth=".\\data\\"):

    
    for i in range(1,10):
        ## Make sure directory contains folders 1-9 and train/test folders
        Path(pth + str(i)).mkdir(parents=True, exist_ok=True)

        # generate <num> images and save them in train folder
        for j in range(num):
            fnt = np.random.choice(glob(".\\sudoku\\fonts\\*"))
            fnt = ImageFont.truetype(font=fnt, size = np.random.randint(30,35))
            img = Image.new('L', (32, 32), color='white')
            d = ImageDraw.Draw(img)
            d.text((8,0),str(i), font=fnt)
            path = pth + str(i) + '\\'
            img.save(path + str(j) + '.png')
        

if __name__ == "__main__":
    generate()


