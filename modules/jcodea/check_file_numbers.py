#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from PIL import Image
# import arrow
import time
from pathlib import Path


dpath = Path('/project/data/sdata/Image/face')


for fold in range(15):
    foldstr = f'fold{fold:0>2}'
    print(fold, foldstr)


for fold in range(15):
    foldstr = f'fold{fold:0>2}'
    pth = dpath/foldstr
    jfiles = pth.glob('*.json')
    mfiles = pth.glob('*.jpeg')
    
#     print(fold, foldstr, pth)
    print(fold, foldstr, len(list(jfiles)),len(list(mfiles) ))
#     time.sleep()


import pkg_resources

