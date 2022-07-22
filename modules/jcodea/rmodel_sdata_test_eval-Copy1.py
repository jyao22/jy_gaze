#!/usr/bin/env python
# coding: utf-8

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
import wandb

import datasets
# from utils import select_device, natural_keys, gazeto3d, angular, getArch
from utils import select_device, natural_keys, gazeto3d, angular, getArch
from model import L2CS


# check if we have the correct number of checkpoint files 
ppath ='/project/results/output/snapshots/' 
for fold in range(15):
    foldstr = f"fold{fold:0>2}"
    cpath =os.path.join(ppath, foldstr)
    files = os.listdir(cpath)
    print(len(files), end=" ")


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


class Nothing:
    pass
args = Nothing()
args.gazeMpiimage_dir = '/project/data/sdata/Image'  # syn data 
args.gazeMpiilabel_dir = '/project/data/sdata/Label'  # syn label
args.output = '/project/results/output/snapshots/'  # real model
args.dataset = 'mpiigaze'
args.snapshot='/project/results/output/snapshots/'  # real data model
args.evalpath = '/project/results/rsoutput/evaluation/'
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


get_ipython().system('ls ../results')


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

model_used=getArch(arch, bins)  #resnet50 and 28 bins


# fold=2
# folder = os.listdir(os.path.join(snapshot_path, "fold" + f'{fold:0>2}'))
# folder.sort(key=natural_keys)
# folder.pop(-1)  #remove the tensorboard file
# # print(folder)
# epochs = folder[3]
# os.path.join(snapshot_path+"fold"+f'{fold:0>2}', epochs)


# print(evalpath, snapshot_path)


# print(data_set)


get_ipython().run_cell_magic('time', '', '\nims_errs = {}\n\nfor fold in range(15):\n    print(f"fold={fold}")\n    \n    now = datetime.utcnow()\n    now = now.astimezone(timezone(\'US/Pacific\'))\n    date_format=\'%m/%d/%Y %H:%M:%S\'\n    now = now.strftime(date_format)\n    \n    print(args.gazeMpiilabel_dir)\n    folder = os.listdir(args.gazeMpiilabel_dir)\n    folder.sort()  #individual label files\n    testlabelpathcombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder] \n    \n    gaze_dataset=datasets.Mpiigaze(testlabelpathcombined, args.gazeMpiimage_dir, transformations, False, angle, fold)\n\n    test_loader = torch.utils.data.DataLoader(\n        dataset=gaze_dataset,\n        batch_size=batch_size,\n        shuffle=False,\n        num_workers=4,\n        pin_memory=True)\n\n    fold_path = os.path.join(evalpath, \'fold\' + f\'{fold:0>2}\'+\'/\')  #for evaluation\n#     print(f"fold_path is {fold_path}")\n    if not os.path.exists(fold_path):\n        os.makedirs(fold_path)\n\n#     if not os.path.exists(os.path.join(evalpath, f"fold"+str(fold))):\n#         os.makedirs(os.path.join(evalpath, f"fold"+str(fold)))\n\n    # list all epochs for testing\n    folder = os.listdir(os.path.join(snapshot_path, "fold" + f\'{fold:0>2}\'))\n    folder.sort(key=natural_keys)\n    folder.pop(-1)  #remove the tensorboard file, now all snapshot files\n#     print(f"folder={folder}")  #contains all the checkpoint files\n                    \n\n    softmax = nn.Softmax(dim=1)\n#     with open(os.path.join(evalpath, os.path.join("fold"+f\'{fold:0>2}\', data_set+".log")), \'w\') as outfile:\n        \n    configuration = (f"\\ntest configuration equal gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\\n"\n                     f"Start testing dataset={data_set}, FOLD={fold} --{now}---------")\n    print(configuration)\n\n#     outfile.write(configuration)\n    epoch_list=[]\n    avg_MAE=[]\n    for epochs in folder: \n        x = \'\'.join(filter(lambda i: i.isdigit(), epochs))\n        x = int(x)\n        if x != 14:\n            continue\n#         print(f"epochs={epochs}")\n        model=model_used\n        checkpoint = torch.load(os.path.join(snapshot_path+"fold"+f\'{fold:0>2}\', epochs))\n#         print(f"checkpoint={checkpoint}")\n        saved_state_dict = checkpoint[\'model_state_dict\']\n        model= nn.DataParallel(model,device_ids=[0])\n        model.load_state_dict(saved_state_dict)\n        model.cuda(gpu)\n        model.eval()\n        total = 0\n        idx_tensor = [idx for idx in range(28)]\n        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)\n        avg_error = .0\n        \n        print(f"epochs={epochs}")\n        \n        with torch.no_grad():\n            for j, (images, labels, cont_labels, name) in enumerate(test_loader):\n                images = Variable(images).cuda(gpu)\n                total += cont_labels.size(0)\n\n                label_pitch = cont_labels[:,0].float()*np.pi/180\n                label_yaw = cont_labels[:,1].float()*np.pi/180\n\n                gaze_pitch, gaze_yaw = model(images)\n\n                # Binned predictions\n                _, pitch_bpred = torch.max(gaze_pitch.data, 1)\n                _, yaw_bpred = torch.max(gaze_yaw.data, 1)\n\n                # Continuous predictions\n                pitch_predicted = softmax(gaze_pitch)\n                yaw_predicted = softmax(gaze_yaw)\n\n                # mapping from binned (0 to 28) to angels (-42 to 42)                \n                pitch_predicted = \\\n                    torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42\n                yaw_predicted = \\\n                    torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42\n\n                pitch_predicted = pitch_predicted*np.pi/180\n                yaw_predicted = yaw_predicted*np.pi/180\n\n#                 print(f"name={name}")\n                errors = []\n                for idx, (p,y,pl,yl) in enumerate(zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw)):\n                    angular_error = angular(gazeto3d([p,y]), gazeto3d([pl,yl]))\n#                         print(f"type OF angular error {type(angular_error)}")\n#                     print(f"angular error={angular_error}")\n                    avg_error += angular_error\n                    angular_error = angular_error.item()  #numpy float to float\n                    errors.append(angular_error)\n                tdict = dict(zip(name, errors))\n                ims_errs.update(tdict)\n        epoch_list.append(x)')


images_errors = dict(sorted(ims_errs.items(), key=lambda item: item[1], reverse=True))


20*15/60


ix=0
for k in images_errors:
    if ix >3:
        break
    print(type(k), k)
    print(type(images_errors[k]), images_errors[k])
    ix += 1


errors = np.array(list(images_errors.values()))


plt.hist(errors)


mean_error = errors.mean()
print(mean_error)


ix = 0
for k, v in images_errors.items():
    if ix >40:
        break
    print(k, v)
    ix += 1




