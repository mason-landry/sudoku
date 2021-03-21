import numpy as np
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from pathlib import Path
 
digit = np.random.randint(1,10)
fnt = np.random.choice(glob(".\\fonts\\*"))
fnt = ImageFont.truetype(font=fnt, size = np.random.randint(30,50))
img = Image.new('RGB', (60, 60), color = 'white')
d = ImageDraw.Draw(img)
d.text((20,10),str(digit), font=fnt,  fill=(0,0,0))
img.save('img.png')

def generate(num = 1000, split=[0.9,0.1], pth=".\\data\\"):

    
    for i in range(1,10):
        ## Make sure directory contains folders 1-9 and train/test folders
        Path(pth + '\\train\\' +  str(i)).mkdir(parents=True, exist_ok=True)
        Path(pth + '\\test\\' + str(i)).mkdir(parents=True, exist_ok=True)

        # generate <num> images and save them in train folder
        for j in range(int(np.ceil(num*split[0]))):
            fnt = np.random.choice(glob(".\\sudoku\\fonts\\*"))
            fnt = ImageFont.truetype(font=fnt, size = np.random.randint(30,50))
            img = Image.new('RGB', (60, 60), color = 'white')
            d = ImageDraw.Draw(img)
            d.text((20,10),str(i), font=fnt,  fill=(0,0,0))
            path = pth + '\\train\\' + str(i) + '\\'
            img.save(path + str(j) + '.png')

            # generate <num> images and save them in test folder. 
        for j in range(int(np.ceil(num*split[1]))):
            fnt = np.random.choice(glob(".\\sudoku\\fonts\\*"))
            fnt = ImageFont.truetype(font=fnt, size = np.random.randint(30,50))
            img = Image.new('RGB', (60, 60), color = 'white')
            d = ImageDraw.Draw(img)
            d.text((20,10),str(i), font=fnt,  fill=(0,0,0))
            path = pth + '\\test\\' + str(i) + '\\'
            img.save(path + str(j) + '.png')
        

if __name__ == "__main__":
    generate()


