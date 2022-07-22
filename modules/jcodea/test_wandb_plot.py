#!/usr/bin/env python
# coding: utf-8

import wandb
import random
import math


# Start a new run
run = wandb.init(project='custom-charts')
offset = random.random()


# At each time step in the model training loop
for run_step in range(20):

  # Log basic experiment metrics, which show up as standard line plots in the UI
  wandb.log({
      "acc": math.log(1 + random.random() + run_step) + offset,
      "val_acc": math.log(1 + random.random() + run_step) + offset * random.random(),
  }, commit=False)

  # Set up data to log in custom charts
  data = []
  for i in range(100):
    data.append([i, random.random() + math.log(1 + i) + offset + random.random()])
  
  # Create a table with the columns to plot
  table = wandb.Table(data=data, columns=["step", "height"])

  # Use the table to populate various custom charts
  line_plot = wandb.plot.line(table, x='step', y='height', title='Line Plot')
  histogram = wandb.plot.histogram(table, value='height', title='Histogram')
  scatter = wandb.plot.scatter(table, x='step', y='height', title='Scatter Plot')
  
  # Log custom tables, which will show up in customizable charts in the UI
  wandb.log({'line_1': line_plot, 
             'histogram_1': histogram, 
             'scatter_1': scatter})

# Finally, end the run. We only need this ine in Jupyter notebooks.
run.finish()


import math
import numpy as np
from matplotlib import pyplot as plt


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

