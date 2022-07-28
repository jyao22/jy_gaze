#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from PIL import Image
from pytz import timezone
from pathlib import Path


sys.path.append('/project/modules/jmodules')
from jutils import SynJSON as SJ, get_now


#source data path
sdatapath = Path('/project/data/download/')
# destination data path
ddatapath = Path('/project/data/sdata3')
labelpath = ddatapath/'Label'
imagepath = ddatapath/'Image'
leftpath = imagepath/'left'
rightpath = imagepath/'right'
facepath = imagepath/'face'
os.makedirs(labelpath, exist_ok=True)
os.makedirs(facepath, exist_ok=True)
os.makedirs(leftpath, exist_ok=True)
os.makedirs(rightpath, exist_ok=True)
label_title = 'Face Left Right Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin'


final_size = 224
ratio = 1.0
half_size = int(final_size*0.5*ratio)
now=get_now()
project = 'create_image_label'
wandb.init(project=propject, name='process_images_labels')
for nf, fold in enumerate(sdatapath.glob('fold*')):
    print(nf, fold, type(fold))
    if nf < 2:
        label_file = 'test.label'
    else:
        label_file = 'train.label'
    label_file = labelpath/label_file
    print(label_file)
    #source
    sfoldpath = fold
    #destination
    if nf < 2 :
        dfoldpath = facepath/'test'
    else:
        dfoldpath = facepath/'train' 
#     dfoldpath = facepath/fold.name
#     print(f'destination dfoldpath = {dfoldpath}')
    os.makedirs(dfoldpath, exist_ok=True)
    
    jfiles = os.listdir(sfoldpath)
    jfiles = [f for f in jfiles if f.endswith('json')]
    jfiles.sort()
    print(f"fold {fold}, n_jfiles={len(jfiles)}")
    now=get_now()
    print(now)
    lf = open(label_file, "a")
    total = 0
    for i, jfile in enumerate(jfiles):
        
        imfile = jfile.replace('info.json', 'rgb.png')
        jfile = sfoldpath/jfile
        imfile = sfoldpath/imfile
        assert imfile.is_file()
        assert jfile.is_file()
        
        if (i+1)%500==0:
            print(f'{i+1}', end=' ')
#             print(f'imfile={imfile}', end=' ')
#             print(f'jfile={imfile}')
        # do the image file
        sj = SJ(jfile)
        
        [pitchl, yawl], [pitchr, yawr] = sj.pitchyaw2d(radian=False, average=False)
        if abs(pitchr-pitchl) > 5.0 or abs(yawr-yawl) > 10.0:
                continue
        
        
        
        face_center = sj.face_center()
        im = Image.open(imfile)
#         display(im)
        assert im.size[0] == im.size[1]
        center = face_center*im.size[0]
        center = center.astype(int)
        crop_position = np.concatenate([center-half_size, center+half_size])
        m1 = im.crop(crop_position)
        m1 = m1.resize((final_size, final_size))
        dimfilename = fold.name+'_'+str(i+1)+'.jpeg'
#         print(f'dimfilename={dimfilename}')
        dimfile = dfoldpath/dimfilename
#         print(f'dimfile={dimfile}')
        m1.save(dimfile)
 
        #write label:
#         face = '/'.join(dimfile.split('/')[-3:])
        face = dimfile.relative_to('/project/data/sdata3/Image/').as_posix()
        left = 'NA'
        right = 'NA'
#         origin = '/'.join(imfile.split('/')[-2:])
        origin = imfile.relative_to('/project/data/download/').as_posix()
#         print(f'face={face}')
#         print(f'origin={origin}')
#         pitch, yaw = sj.pitch_yaw()
        pitch, yaw = sj.pitchyaw2d()  #in radian and average
        whicheye = 'NA'
        d3gaze = '0.0,0.0,0.0'
        d3head = '0.0,0.0,0.0'
        d2gaze = f'{pitch},{yaw}'
        d2head = '0.0,0.0'
        rmat=smat = '1.0,1.0,1.0'
        gazeorigin = '0.0,0.0,0.0'
        to_print = ' '.join([face, left, right, origin, whicheye, d3gaze, d3head, d2gaze, d2head, rmat, smat, gazeorigin])
#         print(to_print)
        print(to_print, file=lf) 
        total += 1
    
    lf.close()
    wandb.log({'total':total})
now = get_now()
print(f'ends at: {now}')
        

