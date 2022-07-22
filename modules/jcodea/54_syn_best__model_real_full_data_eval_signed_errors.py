#!/usr/bin/env python
# coding: utf-8

# Use the best model for each checkpoint folder to test maes for all data folds


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


args = argparse.Namespace()
args.gazeMpiimage_dir = '/project/data/Image'  #real data 
args.gazeMpiilabel_dir = '/project/data/Label'  #real label
# args.output = '/project/results/soutput1/snapshots/'
args.dataset = 'mpiigaze'
args.snapshot='/project/results/soutput1/snapshots/'
# args.evalpath = '/project/results/sroutput1/evaluation/'
args.gpu_id = '0,1,2,3'
args.gpu_id = '0'
args.batch_size = 20
args.arch = 'ResNet50'
args.bins=35
args.angle = 180
args.bin_width = 4


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


#read the fold epoch table
dfile = Path(snapshot_path).parent/"syn_syn__all_MAE.npy"
all_MAE = np.load(dfile)
best_epoch = {}
best_epoch_error = {}
print(f"fold   best_epoch error(degree)")
for idx, fdata in enumerate(all_MAE):
#     print(idx, type(idx))
    epoch_min_error = np.argmin(fdata)+1
    print(f'{idx:^4} {np.argmin(fdata)+1:^12} {fdata.min():^11.3f}')
    best_epoch[idx+1] = epoch_min_error
    best_epoch_error[idx+1]=fdata.min()


# check if we have the correct number of checkpoint files 
ppath = snapshot_path
for fold in range(15):
    foldstr = f"fold{fold:0>2}"
    cpath =os.path.join(ppath, foldstr)
    files = os.listdir(cpath)
    print(f'{fold}:{len(files)}',end=" ")


def get_now():
    now = datetime.utcnow()
    now = now.astimezone(timezone('US/Pacific'))
    date_format='%m/%d/%Y %H:%M:%S'
    now = now.strftime(date_format) 
    return now
print(get_now())


get_ipython().run_cell_magic('time', '', '# all_MAE = []\n# tosave={}\nlfolder = os.listdir(gazeMpiilabel_dir)\nlfolder.sort()  #individual label files\ntestlabelpathcombined = [os.path.join(gazeMpiilabel_dir, j) for j in lfolder] \nprint(testlabelpathcombined)\nmodel = model_used\nmodel= nn.DataParallel(model,device_ids=[0])  #important to load state dict\n# for cfold in range(15):  #checkpoint folds\ncfold=0\n\n\ncfoldstr = f"fold{cfold:0>2}"\nprint(f"cfold={cfoldstr}")\nidx = cfold + 1\nb_epoch = best_epoch[idx]\n\ncheckpoint_path = Path(snapshot_path)/cfoldstr\nbest_checkpoint_file = \'epoch_\'+str(b_epoch)+\'.pkl\'\nbest_checkpoint_file = checkpoint_path/best_checkpoint_file\n#     print(best_checkpoint_file, type(best_checkpoint_file))\ncheckpoint = torch.load(best_checkpoint_file)\nsaved_state_dict = checkpoint[\'model_state_dict\']\nmodel.load_state_dict(saved_state_dict)\n\nmodel.cuda(gpu)\nmodel.eval()\nidx_tensor = [idx for idx in range(35)]\nidx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)\nsoftmax = nn.Softmax(dim=1)\nbest_model_maes =[]  # error for each data fold\n\n\nall_pitches=[]\nall_pitch_errors =[]\nall_yaws=[]\nall_yaw_errors =[]\n\nfor dfold in range(15):  #data folds\n    gaze_dataset=datasets.Mpiigaze(testlabelpathcombined, gazeMpiimage_dir, transformations, False, angle, dfold)\n    test_loader = torch.utils.data.DataLoader(\n        dataset=gaze_dataset,\n        batch_size=batch_size,\n        shuffle=True,\n        num_workers=4,\n        pin_memory=True)\n    now = get_now()\n    configuration = f"\\ntest configuration: gpu_id={gpu}, batch_size={batch_size}\\n"\n    configuration += f"model_arch={arch} Start testing dataset={data_set}\\n"\n    configuration +=  f"cFOLD={cfold} dFOLD={dfold}--{now}----"\n    print(configuration)\n\n#     total = 0  \n#     avg_error = .0\n    with torch.no_grad():\n        for j, (images, labels, cont_labels, name) in enumerate(test_loader):\n            images = Variable(images).cuda(gpu)\n#             total += cont_labels.size(0)\n\n            label_pitch = cont_labels[:,1].float()*np.pi/180\n            label_yaw = cont_labels[:,0].float()*np.pi/180\n            label_yaw = label_yaw*(-1.0)\n\n            gaze_pitch, gaze_yaw = model(images)\n\n            # Binned predictions\n            _, pitch_bpred = torch.max(gaze_pitch.data, 1)\n            _, yaw_bpred = torch.max(gaze_yaw.data, 1)\n\n            # Continuous predictions\n            pitch_predicted = softmax(gaze_pitch)\n            yaw_predicted = softmax(gaze_yaw)\n\n            # mapping from binned (0 to 28) to angels (-42 to 42)                \n            pitch_predicted = \\\n                torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 52\n            yaw_predicted = \\\n                torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 52\n\n            label_yaw = label_yaw*180/np.pi\n            label_pitch = label_pitch*180/np.pi\n            \n#             pitch_predicted = pitch_predicted\n#             yaw_predicted = yaw_predicted\n            \n            pitch_errors = pitch_predicted - label_pitch\n            yaw_errors = yaw_predicted - label_yaw\n            \n            all_yaws.extend(label_yaw.numpy())\n            all_pitches.extend(label_pitch.numpy())\n            all_yaw_errors.extend(yaw_errors.numpy())\n            all_pitch_errors.extend(pitch_errors.numpy())\n            if j == 2:\n                print(f"pitches:{label_pitch}")\n                print(f"yaws:{label_yaw}")\n                print(f"pitches_predicted:{pitch_predicted}")\n                print(f"yaws_predicted:{yaw_predicted}")\n                print(f"pitches_errors:{pitch_errors}")\n                print(f"yaws_errors:{yaw_errors}")')


all_yaws = np.array(all_yaws)
all_pitches = np.array(all_pitches)
all_yaw_errors = np.array(all_yaw_errors)
all_pitch_errors = np.array(all_pitch_errors)


# frun = wandb.init(project="54_best_smodel_all_rdata_sign_errors")
# frun.name = f'signed_errors_best_smodel_all_rdata'


plt.hist(all_yaw_errors, bins=40)


with wandb.init(project="54 best smodel all rdata signed errors") as run:
    plt.hist(all_yaw_errors, bins=40)
    plt.title("signed yaw error distribution: best syn model all_real data")
    plt.xlabel("degree")
    run.log({"data":wandb.Image(plt)})


with wandb.init(project="54 best smodel all rdata signed errors") as run:
    plt.hist(all_pitch_errors, bins=40)
    plt.title("signed pitch error distribution: best syn model all_real data")
    plt.xlabel("degree")
    run.log({"data":wandb.Image(plt)})





with wandb.init(project="54 best smodel all rdata signed errors") as run:
    data = [[x, y] for (x, y) in zip(all_yaws, all_yaw_errors)]
    table = wandb.Table(data=data, columns = ["truth_degree", "error_degree"])
    wandb.log({"yaw_error_scatter" : wandb.plot.scatter(table, "truth_degree", "error_degree", title="pitch error vs ground truth")})


with wandb.init(project="54 best smodel all rdata signed errors") as run:
    data = [[x, y] for (x, y) in zip(all_pitches, all_pitch_errors)]
    table = wandb.Table(data=data, columns = ["truth_degree", "error_degree"])
    wandb.log({"pitch_error_scatter" : wandb.plot.scatter(table, "truth_degree", "error_degree", title="yaw error vs ground truth")})

