#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import glob
# import seaborn as sns
from PIL import Image
from PIL import ImageStat
from pathlib import Path
import matplotlib.pyplot as plt


# def calculate_brightness(image):
#     greyscale_image = image.convert('L')
#     histogram = greyscale_image.histogram()
#     pixels = sum(histogram)
#     brightness = scale = len(histogram)

#     for index in range(0, scale):
#         ratio = histogram[index] / pixels
#         brightness += ratio * (-scale + index)

#     return 1 if brightness == 255 else brightness / scale


impath = Path('../data/Image')


def brightness(im_file ):
    img = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(img)
    return stat.rms[0]/255.0


get_ipython().system('ls ../data/Image/p02/face/28.jpg')


bright_imgf = Path('../data/Image/p02/face/28.jpg')


dark_imgf = Path('../data/Image/p02/face/42.jpg')


print(brightness(dark_imgf)), print(brightness(bright_imgf))


get_ipython().run_cell_magic('time', '', 'bnesses =[]\nfor fold in sorted(impath.iterdir()):\n    print(f"fold={fold}")\n    for faced in fold.iterdir():\n        if faced.name == \'face\':\n            for imfile in faced.iterdir():\n                try:\n                    bness = brightness(imfile)\n                    bnesses.append(bness)\n#                     print(bness)\n                except:\n                    continue')


bnesses = np.array(bnesses)
plt.hist(bnesses, bins=100)
plt.title("brightness distribution of real face iamges")
plt.xlabel("brightness")


sys.exit()

