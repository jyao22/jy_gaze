#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from math import atan2, asin
from pathlib import Path
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import matplotlib.pyplot as plt
import wandb


sys.path.append('/project/modules/jmodules')
from jutils import SynJSON as SJ


dpath = Path('/project/data/download/')
project = 'sdata3_sample_angle_check'


# %%time
pitches = []
yaws = []
pitchesl =[]
yawsl=[]
pitchesr =[]
yawsr = []
iffy_pitches = 0
iffy_yaws = 0
for nf, fold in enumerate(dpath.glob('fold*')):
    if nf>2:
        break 
    if nf >0:
        print(f"\nfold={fold}")
    else:
        print(f'fold={fold}')
    dpath = dpath/fold
    j =0
    for jfile in dpath.glob('*.json'):
#         print(jfile)
        j += 1
        if j % 600== 0:
            print(f"{j}", end=' ')
        sj = SJ(jfile)
        pitch, yaw = sj.pitchyaw2d(radian=False, average=True)
        [pitchl, yawl], [pitchr, yawr] = sj.pitchyaw2d(radian=False, average=False)
        pitches.append(pitch)
        yaws.append(yaw)
        pitchesl.append(pitchl)
        pitchesr.append(pitchr)
        yawsl.append(yawl)
        yawsr.append(yawr)
        if abs(yawl-yawr)>3.0:
            iffy_yaws += 1
#             print(yawl, yawr)
        if abs(pitchl - pitchr)>3.0:
            iffy_pitches += 1


num_bins=50
fig1, ax = plt.subplots()
n, bins, patches = ax.hist(pitches, num_bins, 
                            density = 1, 
                            color ='g',
                            alpha = 0.7)
ax.set_title('distribution of pitches')
ax.set_xlabel('degree')
ax.set_xlim(-23, 7)


num_bins=50
n, bins, patches = plt.hist(pitches, num_bins, 
                            density = 1, 
                            color ='g',
                            alpha = 0.7)
plt.title('distribution of pitches')
plt.xlabel('degree')
plt.xlim(-23, 7)

with wandb.init(project=project, name='sample_pitches_histogram') as run:
    run.log({"data3":wandb.Image(plt)})


num_bins=50
n, bins, patches = plt.hist(yaws, num_bins, 
                            density = 1, 
                            color ='m',
                            alpha = 0.7)
plt.title('distribution of yaws')
plt.xlabel('degree')
plt.xlim(-25, 25)

with wandb.init(project=project, name='sample_yaws_histogram') as run:
    run.log({"data3":wandb.Image(plt)})


plt.scatter(x=pitchesl, y=pitchesr)
plt.title('pitches of left and right eyes')
plt.xlabel('pitch: left eye (degree)')
plt.ylabel('pitch: right eye (degree)')
plt.grid(visible=True)
with wandb.init(project=project, name='sample_pitches_corr') as run:
    run.log({"data4":wandb.Image(plt)})


plt.scatter(x=yawsl, y=yawsr)
plt.title('yaws of left and right eyes')
plt.xlabel('yaw: left eye (degree)')
plt.ylabel('yaw: right eye (degree)')
plt.grid(visible=True)

with wandb.init(project=project, name='sample_yaws_corr') as run:
    run.log({"data4":wandb.Image(plt)})


adiffs = []
for yawl, yawr in zip(yawsl, yawsr):
    adiff = abs(yawl-yawr)
#     print(yawl, yawr, diff)
    adiffs.append(adiff)


num_bins=1000
n, bins, patches = plt.hist(adiffs, num_bins, 
                            density = 1, 
                            color ='b',
                            alpha = 0.7)
plt.title('distribution of yaw absolute differences')
plt.xlabel('degree difference')
plt.xlim(3, 7)
with wandb.init(project=project, name='absolute diffs yaw right and left') as run:
    run.log({"data5":wandb.Image(plt)})


diffs = []
for yawl, yawr in zip(yawsl, yawsr):
    diff = yawl-yawr
    if yawr > yawl:
        print(yawl, yawr, diff)
    diffs.append(diff)


num_bins=1000
n, bins, patches = plt.hist(diffs, num_bins, 
                            density = 1, 
                            color ='g',
                            alpha = 0.7)
plt.title('distribution of yaw differences')
plt.xlabel('degree difference')
plt.xlim(3.0, 6)

with wandb.init(project=project, name='diffs yaw right and left') as run:
    run.log({"data5":wandb.Image(plt)})

