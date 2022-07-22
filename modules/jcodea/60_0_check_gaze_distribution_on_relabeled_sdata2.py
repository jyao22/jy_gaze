#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import wandb
# import seaborn as sns


label_dir = "../data/sdata2/Label"


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


wandb.init(project="angle_distribution_sdata2")


with wandb.init(project="angle_distribution_sdata2") as run:
    num_bins=50
    n, bins, patches = plt.hist(pitches, num_bins, 
                                density = 1, 
                                color ='green',
                                alpha = 0.7)
    plt.title('distribution of pitches')
    plt.xlabel('degree')
    run.log({"data":wandb.Image(plt)})

    # num_bins=50
    n, bins, patches = plt.hist(yaws, num_bins, 
                                density = 1, 
                                color ='green',
                                alpha = 0.7)
    plt.title('distribution of yaws')
    plt.xlabel('degree')
    
    run.log({"data":wandb.Image(plt)})


num_bins=50
n, bins, patches = plt.hist(pitches, num_bins, 
                            density = 1, 
                            color ='green',
                            alpha = 0.7)
plt.title('distribution of pitches')
plt.xlabel('degree')


num_bins=50
n, bins, patches = plt.hist(yaws, num_bins, 
                            density = 1, 
                            color ='green',
                            alpha = 0.7)
plt.title('distribution of yaws')
plt.xlabel('degree')







