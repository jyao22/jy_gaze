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
dfile = Path(snapshot_path).parent/"syn_syn_all_MAE.npy"
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


get_ipython().run_cell_magic('time', '', 'all_MAE = []\n# tosave={}\nlfolder = os.listdir(gazeMpiilabel_dir)\nlfolder.sort()  #individual label files\ntestlabelpathcombined = [os.path.join(gazeMpiilabel_dir, j) for j in lfolder] \nprint(testlabelpathcombined)\nmodel = model_used\nmodel= nn.DataParallel(model,device_ids=[0])  #important to load state dict\nfor cfold in range(15):  #checkpoint folds\n    frun = wandb.init(project="53_best_checkpoint_model_all_real_data_folds")\n    frun.name = f\'checkpoint fold{cfold}_mae\'\n    \n    cfoldstr = f"fold{cfold:0>2}"\n    print(f"cfold={cfoldstr}")\n    idx = cfold + 1\n    b_epoch = best_epoch[idx]\n    \n    checkpoint_path = Path(snapshot_path)/cfoldstr\n    best_checkpoint_file = \'epoch_\'+str(b_epoch)+\'.pkl\'\n    best_checkpoint_file = checkpoint_path/best_checkpoint_file\n#     print(best_checkpoint_file, type(best_checkpoint_file))\n    checkpoint = torch.load(best_checkpoint_file)\n    saved_state_dict = checkpoint[\'model_state_dict\']\n    model.load_state_dict(saved_state_dict)\n\n    model.cuda(gpu)\n    model.eval()\n    idx_tensor = [idx for idx in range(35)]\n    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)\n    softmax = nn.Softmax(dim=1)\n    best_model_maes =[]  # error for each data fold\n    for dfold in range(15):  #data folds\n        gaze_dataset=datasets.Mpiigaze(testlabelpathcombined, gazeMpiimage_dir, transformations, False, angle, dfold)\n        test_loader = torch.utils.data.DataLoader(\n            dataset=gaze_dataset,\n            batch_size=batch_size,\n            shuffle=True,\n            num_workers=4,\n            pin_memory=True)\n        now = get_now()\n        configuration = f"\\ntest configuration: gpu_id={gpu}, batch_size={batch_size}\\n"\n        configuration += f"model_arch={arch} Start testing dataset={data_set}\\n"\n        configuration +=  f"cFOLD={cfold} dFOLD={dfold}--{now}----"\n        print(configuration)\n        \n        total = 0  \n        avg_error = .0\n        with torch.no_grad():\n            for j, (images, labels, cont_labels, name) in enumerate(test_loader):\n                images = Variable(images).cuda(gpu)\n                total += cont_labels.size(0)\n\n                label_pitch = cont_labels[:,0].float()*np.pi/180\n                label_yaw = cont_labels[:,1].float()*np.pi/180\n\n                gaze_pitch, gaze_yaw = model(images)\n\n                # Binned predictions\n                _, pitch_bpred = torch.max(gaze_pitch.data, 1)\n                _, yaw_bpred = torch.max(gaze_yaw.data, 1)\n\n                # Continuous predictions\n                pitch_predicted = softmax(gaze_pitch)\n                yaw_predicted = softmax(gaze_yaw)\n\n                # mapping from binned (0 to 28) to angels (-42 to 42)                \n                pitch_predicted = \\\n                    torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 52\n                yaw_predicted = \\\n                    torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 52\n\n                pitch_predicted = pitch_predicted*np.pi/180\n                yaw_predicted = yaw_predicted*np.pi/180\n\n                for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):\n                    pl, yl = yl, pl*(-1.0)\n                    avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl])) #accumulate over all batches\n                    \n        mean_mae = avg_error/total  \n        now = get_now()\n        msg = f"[cfold{cfold}---dfold {dfold}] Total Num:{total}, MAE:{mean_mae}  {now}"\n#         outfile.write(loger)\n        print(msg)\n        best_model_maes.append(mean_mae)  # a row of maes for each data fold\n        # log the mean error for the data fold for diagnostic purposes\n        frun.log({\'data fold\': dfold, f\'dfold_{dfold}_avg_mae\':mean_mae}, step=dfold)\n    bestmodel_mean = np.mean(best_model_maes)  # the mean mae for the model  \n    msg = f"[cfold{cfold}--] MAE for this fold best model:{bestmodel_mean}  {now}"\n    print(msg)\n    all_MAE.append(best_model_maes)  # append the whole row of data fold maes for each best fold model\n    # log for each best model\n    ')


all_MAE=np.array(all_MAE)


print(all_MAE.shape)
print("data fold means, which data fold is bad or good:\n",all_MAE.mean(axis=0))
print("model means, which model ot choose:\n", all_MAE.mean(axis=1))
# print("data fold min:", all_MAE.min(axis=0))
# print("model min:", all_MAE.min(axis=1))
# print("data fold max:", all_MAE.max(axis=0))
# print("model max:", all_MAE.max(axis=1))





save_file = Path(snapshot_path).parent/"cfold_dfold_all_MAE.npy"
with open(save_file, 'wb') as f:
    np.save(f, all_MAE)

