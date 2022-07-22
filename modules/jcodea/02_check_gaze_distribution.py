#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import wandb

label_dir = "../data/Label"

dirs = os.listdir(label_dir)
# print(dirs)

label_files = glob.glob(label_dir+'/p*')
label_files.sort()
# print(label_files)

pitches = []
yaws = []
for lfile in label_files:
    with open(lfile) as f:
        lines = list(f)
        for line in lines[1:]:
            line = line.strip().split(" ")
            gaze2d = line[7]
            label = np.array(gaze2d.split(",")).astype("float")
#             label = torch.from_numpy(label).type(torch.FloatTensor)
            pitch = label[0]* 180 / np.pi
            yaw = label[1]* 180 / np.pi
            pitches.append(pitch)
            yaws.append(yaw)
#             print(pitch, yaw)

pitches = np.array(pitches)
yaws = np.array(yaws)

plt.hist(pitches,bins=50)
plt.title("pose pitch distribution")
plt.xlabel("degree")

with wandb.init(project='real data pose and gaze distributions') as run:
    run.log({"data": wandb.Image(plt)})

plt.cla()    
plt.hist(yaws,bins=50)
plt.title("pose yaw distribution")
plt.xlabel("degree")

with wandb.init(project='real data pose and gaze distributions') as run:
    run.log({"data": wandb.Image(plt)})

