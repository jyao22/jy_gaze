#!/usr/bin/env python
# coding: utf-8

# import wandb


# wandb.init(project='testgaze')


import math
import numpy as np
from matplotlib import pyplot as plt


ims_errs = {}
abcs = list("ABCDE")
vs = [3,1,5,4,9]


saved = dict(zip(abcs, vs))


saved


ims_errs.update(saved)


print(ims_errs)


for fold  in range(3):
    c = 10.0*np.random.random()
    xs = []
    ys =[]
    for epoch in range(60):
        loss = c-c/(1+math.exp(-0.1*epoch))
        wandb.log({'epoch':epoch, 'loss':loss})
        xs.append(epoch)
        ys.append(loss)
    plt.plot(xs, ys)


sys.exit()


# wandb.log({"x": 6, "y": 10})


# wandb.log({"x": 8, "y": 12})


# import numpy as np


get_ipython().system('ls ..')


3*60/60


get_ipython().system('df -m . ')


yaw = 25
pitch = -5


bins = np.array(range(-42, 42, 2))
binned_pose = np.digitize([pitch, yaw], bins) - 1


print(bins, len(bins))


print(binned_pose)


import timeit
import time


for fold in range(5):
    print(f"fold{fold:0>2}")


y="price100.txt"
x = ''.join(filter(lambda i: i.isdigit(), y))
print(x)





from datetime import datetime
from pytz import timezone
import pytz

date_format='%m/%d/%Y %H:%M:%S %Z'
date = datetime.now(tz=pytz.utc)
print('Current date & time is:', date.strftime(date_format))

date = date.astimezone(timezone('US/Pacific'))

print('Local date & time is  :', date.strftime(date_format))


now = datetime.utcnow()
now = now.astimezone(timezone('US/Pacific'))
# print(now)
# date_format='%m_%d_%Y_%H_%M_%S'
date_format='%m/%d/%Y %H:%M:%S'
now = now.strftime(date_format)
print(now)

    
    


get_ipython().system('ls ../results/output/snapshots/')


get_ipython().run_line_magic('tensorboard', "--logdir='/project/results/output/snapshots/fold13/'")





np.array([98,94.5,102,92,102,91]).mean()

