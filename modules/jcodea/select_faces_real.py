#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
from PIL import Image


get_ipython().system('ls ../data/Label')


ipath = Path('../data/Image/')
rlabelpath = Path('../data/Label')





def check_good(pitch, yaw, p, y):
    if abs(pitch-p)<0.5 and abs(yaw-y) <0.5:
        return True
    else:
        return False


nf =0
for rlfile in rlabelpath.glob('*.label'):
    nf += 1
#     if nf >2:
#         break
    with open(rlfile) as rf:
        lines = rf.readlines()
        for line in lines[1:]:
            gaze2d = line.strip().split(" ")[7]
            face = line.strip().split(" ")[0]
            label = np.array(gaze2d.split(",")).astype("float")
            pitch = label[0]* 180 / np.pi
            yaw = label[1]* 180 / np.pi
            if check_good(pitch, yaw, 0, 3):
#                 print(line)
                ifile = ipath/face
                print(pitch, yaw, face, ifile)
                
                
                
                


# pitch = -10, yaw = 0)
ifile = Path('../data/Image/p02/face/30.jpg')
im = Image.open(ifile)
im.show(0)


# (pitch=10, yaw = 0)
ifile = Path('../data/Image/p02/face/395.jpg')
im = Image.open(ifile)
im.show(0)


# (pitch=0, yaw = -10)
ifile = Path('../data/Image/p02/face/1057.jpg')
im = Image.open(ifile)
im.show(0)


# (pitch=0, yaw = 3.4)
ifile = Path('../data/Image/p14/face/253.jpg')
im = Image.open(ifile)
im.show(0)





ipath = Path('../data/sdata/Image/')
rlabelpath = Path('../data/sdata/Label')


nf =0
for rlfile in rlabelpath.glob('*.label'):
    nf += 1
#     if nf >2:
#         break
    with open(rlfile) as rf:
        lines = rf.readlines()
        for line in lines[1:]:
            gaze2d = line.strip().split(" ")[7]
            face = line.strip().split(" ")[0]
            label = np.array(gaze2d.split(",")).astype("float")
            pitch = label[0]* 180 / np.pi
            yaw = label[1]* 180 / np.pi
            if check_good(pitch, yaw, 0, -10):
#                 print(line)
                ifile = ipath/face
                print(pitch, yaw, face, ifile)
                


# pitch = -10, yaw = 0)
ifile = Path('../data/sdata/Image/face/fold08/1840.jpeg')
im = Image.open(ifile)
im.show(0)


# pitch = 10, yaw = 0)
ifile = Path('../data/sdata/Image/face/fold00/1065.jpeg')
im = Image.open(ifile)
im.show(0)


# pitch = 0, yaw = 10)
ifile = Path('../data/sdata/Image/face/fold05/1547.jpeg')
im = Image.open(ifile)
im.show()


# pitch = 0, yaw = -10)
ifile = Path('../data/sdata/Image/face/fold14/2847.jpeg')
im = Image.open(ifile)
im.show()

