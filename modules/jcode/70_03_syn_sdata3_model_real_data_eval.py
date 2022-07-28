#!/usr/bin/env python
# coding: utf-8

# Use the best model for each checkpoint folder to test maes for all data folds


import sys
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import wandb

import datasets
# from utils import select_device, natural_keys, gazeto3d, angular, getArch
from utils import select_device, natural_keys, gazeto3d, angular, getArch
from model import L2CS
sys.path.append('/project/modules/jmodules')
from jutils import get_now


get_ipython().system('pwd')


get_ipython().system('nvidia-smi')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet')
#     # Gaze360
#     parser.add_argument(
#         '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
#         default='datasets/Gaze360/Image', type=str)
#     parser.add_argument(
#         '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
#         default='datasets/Gaze360/Label/train.label', type=str)
#     # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='/project/data/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='/project/data/Label', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze',
        default= "mpiigaze", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='/project/results/soutput3/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='/project/results/soutput3/snapshots/', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path to save the evaluation results',
        default='/project/results/soutput3/evaluation/', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=100, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    parser.add_argument(
        '--bins', dest='bins', help='number of angle bins',
        default=28, type=int)
    parser.add_argument(
        '--angle', dest='angle', help='angle limit',
        default=180, type=int)
    parser.add_argument(
        '--bin_width', dest='bin_width', help='width of anlge bins',
        default=4, type=int)
    
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args(['--angle', '180'])
    return args


args=parse_args()


# actually used
batch_size=args.batch_size
arch=args.arch
data_set=args.dataset
# evalpath =args.evalpath
snapshot_path = args.snapshot
bins=args.bins
angle=args.angle
bin_width=args.bin_width
gazeMpiimage_dir = args.gazeMpiimage_dir
gazeMpiilabel_dir=args.gazeMpiilabel_dir


# args = parse_args()
cudnn.enabled = True
gpu = select_device(args.gpu_id, batch_size=args.batch_size)
transformations = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
model_used= getArch(arch, bins)  #resnet50 and 28 bins


spath = Path(snapshot_path)
ckfiles =[]
for filename in sorted(spath.glob('*.pkl'), 
        key=lambda path: int(path.stem.rsplit("_", 1)[1])):
    ckfiles.append(filename)
print(f'number of checkpoint files: {len(ckfiles)}')


#labels
lfolder = os.listdir(gazeMpiilabel_dir)
lfolder.sort()  #individual label files
testlabelpathcombined = [os.path.join(gazeMpiilabel_dir, j) for j in lfolder]
gaze_dataset=datasets.Mpiigaze(testlabelpathcombined, gazeMpiimage_dir, transformations, False, angle, fold=-1)
# print(testlabelpathcombined)

model = model_used
model= nn.DataParallel(model, device_ids=[0])  #important to load state dict

wandb.init(project='70_smodel_sdata3_rdata', name="MAE_epoch_models")
for ckn, ckfile in enumerate(ckfiles):
    print(ckn+1, ckfile)
    
    checkpoint = torch.load(ckfile)
    saved_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()
    idx_tensor = [idx for idx in range(bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    softmax = nn.Softmax(dim=1)
#     best_model_maes =[]  # error for each data fold
    test_loader = torch.utils.data.DataLoader(
            dataset=gaze_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
    now = get_now()
    configuration = f"\ntest configuration: gpu_id={gpu}, batch_size={batch_size}\n"
    configuration += f"model_arch={arch} Start testing dataset={data_set}--{now}--\n"
    print(configuration)
    
    total = 0  
    avg_error = .0
    with torch.no_grad():
        for j, (images, labels, cont_labels, name) in enumerate(test_loader):
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)  #number of labels/images

            label_pitch = cont_labels[:,0].float()*np.pi/180
            label_yaw = cont_labels[:,1].float()*np.pi/180

            gaze_pitch, gaze_yaw = model(images)

            # Binned predictions
            _, pitch_bpred = torch.max(gaze_pitch.data, 1)
            _, yaw_bpred = torch.max(gaze_yaw.data, 1)

            # Continuous predictions
            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)

            # mapping from binned (0 to 28) to angels (-42 to 42)                
            pitch_predicted =  torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42
            yaw_predicted =  torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42

            pitch_predicted = pitch_predicted*np.pi/180
            yaw_predicted = yaw_predicted*np.pi/180

            for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                # pl, yl = yl, pl*(-1.0)
                avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl])) #accumulate over all batches

    mean_mae = avg_error/total  
    now = get_now()
    msg = f"Total Num images Checked:{total}, MAE:{mean_mae}  {now}"
#         outfile.write(loger)
    wandb.log({'Mean_MAE':mean_mae}, step=ckn+1)
    print(msg)
#     best_model_maes.append(mean_mae) 
    

