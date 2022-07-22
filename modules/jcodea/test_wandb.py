#!/usr/bin/env python
# coding: utf-8

import wandb


import math
import numpy as np
# from matplotlib import pyplot as plt


def run_fold(fold, wrun):
    c = 10.0*np.random.random()
    for epoch in range(60):
        loss = c-c/(1+math.exp(-0.1*epoch))
        wrun.log({f'fold_{fold}_loss':loss})


for fold  in range(10):
    run = wandb.init(project='testgaze')
    run_fold(fold, run)
    

