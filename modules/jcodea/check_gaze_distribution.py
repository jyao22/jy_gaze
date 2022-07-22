#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import glob
import seaborn as sns


label_dir = "../data/Label"


dirs = os.listdir(label_dir)
print(dirs)


label_files = glob.glob(label_dir+'/p*')
label_files.sort()
print(label_files)


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





sns.set_theme()
np.random.seed(0)
ax = sns.distplot(pitches)


ax = sns.distplot(yaws)



print(sorted(pitches)[0:5], sorted(pitches)[-5:])
print(f"total number of pitch values: {len(pitches)}")
print(pitches.mean())


print(sorted(yaws)[0:5])
print(sorted(yaws)[-5:])
print(f"total number of yaw values: {len(yaws)}")
print(yaws.mean())

