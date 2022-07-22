#!/usr/bin/env python
# coding: utf-8

import wandb
wandb.init(project="sdata_test_eval")


import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import datasets
# from utils import select_device, natural_keys, gazeto3d, angular, getArch
from utils import select_device, natural_keys, gazeto3d, angular, getArch
from model import L2CS


# check if we have the correct number of checkpoint files 
ppath ='/project/results/soutput/snapshots/' 
for fold in range(15):
    foldstr = f"fold{fold:0>2}"
    cpath =os.path.join(ppath, foldstr)
    files = os.listdir(cpath)
    print(len(files), end=" ")


now = datetime.utcnow()
now = now.astimezone(timezone('US/Pacific'))
date_format='%m/%d/%Y %H:%M:%S'
now = now.strftime(date_format)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
     # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/test.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)
    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='gaze360, mpiigaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path to the folder contains models.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4-lr', type=str)
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path for the output evaluating gaze test.',
        default="evaluation/L2CS-gaze360-_loader-180-4-lr", type=str)
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=100, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args


# def getArch(arch,bins):
#     # Base network structure
#     if arch == 'ResNet18':
#         model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
#     elif arch == 'ResNet34':
#         model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
#     elif arch == 'ResNet101':
#         model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
#     elif arch == 'ResNet152':
#         model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
#     else:
#         if arch != 'ResNet50':
#             print('Invalid value for architecture is passed! '
#                 'The default value of ResNet50 will be used instead!')
#         model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
#     return model


class Nothing:
    pass
args = Nothing()
args.gazeMpiimage_dir = '/project/data/sdata/Image'
args.gazeMpiilabel_dir = '/project/data/sdata/Label'
args.output = '/project/results/soutput/snapshots/'
args.dataset = 'mpiigaze'
args.snapshot='/project/results/soutput/snapshots/'
args.evalpath = '/project/results/soutput/evaluation/'
args.gpu_id = '0,1,2,3'
args.gpu_id = '0'
args.batch_size = 20
args.arch = 'ResNet50'
args.bins=28
args.angle = 180
args.bin_width = 4


batch_size=args.batch_size
arch=args.arch
data_set=args.dataset
evalpath =args.evalpath
snapshot_path = args.snapshot
bins=args.bins
angle=args.angle
bin_width=args.bin_width


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

model_used=getArch(arch, bins)


# fold=2
# folder = os.listdir(os.path.join(snapshot_path, "fold" + f'{fold:0>2}'))
# folder.sort(key=natural_keys)
# folder.pop(-1)  #remove the tensorboard file
# # print(folder)
# epochs = folder[3]
# os.path.join(snapshot_path+"fold"+f'{fold:0>2}', epochs)


# print(evalpath, snapshot_path)


# print(data_set)


get_ipython().run_cell_magic('time', '', 'all_MAE = []\nfor fold in range(15):\n#     print(f"fold={fold}")\n    \n    now = datetime.utcnow()\n    now = now.astimezone(timezone(\'US/Pacific\'))\n    date_format=\'%m/%d/%Y %H:%M:%S\'\n    now = now.strftime(date_format)\n    \n    folder = os.listdir(args.gazeMpiilabel_dir)\n    folder.sort()\n    testlabelpathcombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder] \n    gaze_dataset=datasets.Mpiigaze(testlabelpathcombined,args.gazeMpiimage_dir, transformations, False, angle, fold)\n\n    test_loader = torch.utils.data.DataLoader(\n        dataset=gaze_dataset,\n        batch_size=batch_size,\n        shuffle=True,\n        num_workers=4,\n        pin_memory=True)\n\n    fold_path = os.path.join(evalpath, \'fold\' + f\'{fold:0>2}\'+\'/\')\n#     print(f"fold_path is {fold_path}")\n    if not os.path.exists(fold_path):\n        os.makedirs(fold_path)\n\n#     if not os.path.exists(os.path.join(evalpath, f"fold"+str(fold))):\n#         os.makedirs(os.path.join(evalpath, f"fold"+str(fold)))\n\n    # list all epochs for testing\n    folder = os.listdir(os.path.join(snapshot_path, "fold" + f\'{fold:0>2}\'))\n    folder.sort(key=natural_keys)\n    folder.pop(-1)  #remove the tensorboard file\n#     print(f"folder={folder}")\n                    \n\n    softmax = nn.Softmax(dim=1)\n    with open(os.path.join(evalpath, os.path.join("fold"+f\'{fold:0>2}\', data_set+".log")), \'w\') as outfile:\n        \n        configuration = (f"\\ntest configuration equal gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\\n"\n                         f"Start testing dataset={data_set}, FOLD={fold} --{now}---------")\n        print(configuration)\n        \n        outfile.write(configuration)\n        epoch_list=[]\n        avg_MAE=[]\n        \n        for epochs in folder: \n#             print(f"entering epochs={epochs}")\n            model=model_used\n            checkpoint = torch.load(os.path.join(snapshot_path+"fold"+f\'{fold:0>2}\', epochs))\n            saved_state_dict = checkpoint[\'model_state_dict\']\n            model= nn.DataParallel(model,device_ids=[0])\n            model.load_state_dict(saved_state_dict)\n            model.cuda(gpu)\n            model.eval()\n            total = 0\n            idx_tensor = [idx for idx in range(28)]\n            idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)\n            avg_error = .0\n            with torch.no_grad():\n                for j, (images, labels, cont_labels, name) in enumerate(test_loader):\n                    images = Variable(images).cuda(gpu)\n                    total += cont_labels.size(0)\n\n                    label_pitch = cont_labels[:,0].float()*np.pi/180\n                    label_yaw = cont_labels[:,1].float()*np.pi/180\n\n                    gaze_pitch, gaze_yaw = model(images)\n\n                    # Binned predictions\n                    _, pitch_bpred = torch.max(gaze_pitch.data, 1)\n                    _, yaw_bpred = torch.max(gaze_yaw.data, 1)\n\n                    # Continuous predictions\n                    pitch_predicted = softmax(gaze_pitch)\n                    yaw_predicted = softmax(gaze_yaw)\n\n                    # mapping from binned (0 to 28) to angels (-42 to 42)                \n                    pitch_predicted = \\\n                        torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42\n                    yaw_predicted = \\\n                        torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42\n\n                    pitch_predicted = pitch_predicted*np.pi/180\n                    yaw_predicted = yaw_predicted*np.pi/180\n\n                    for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):\n                        avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))\n\n            x = \'\'.join(filter(lambda i: i.isdigit(), epochs))\n#             print(f"x={x}")\n            epoch_list.append(x)\n            avg_MAE.append(avg_error/total)  \n#             print(f"total={total}")\n            loger = f"[{epochs}---{args.dataset}] Total Num:{total},MAE:{avg_error/total}"\n            outfile.write(loger)\n            print(loger)\n#             print(f"done epoch={epochs}")\n    \n    all_MAE.append(avg_MAE)\n    fig = plt.figure(figsize=(14, 8))        \n    plt.xlabel(\'epoch\')\n    plt.ylabel(\'avg\')\n    plt.title(\'Gaze angular error\')\n    plt.legend()\n    plt.plot(epoch_list, avg_MAE, color=\'k\', label=\'mae\')\n    fig.savefig(os.path.join(evalpath, os.path.join("fold"+f\'{fold:0>2}\', data_set+".png")), format=\'png\')\n    # plt.show() ')


all_MAE=np.array(all_MAE)


print(all_MAE.shape)
print(all_MAE.mean(axis=0))
print(all_MAE.mean(axis=1))
print(all_MAE.mean(axis=0).argmin()+1 ,all_MAE.mean(axis=0).min())

