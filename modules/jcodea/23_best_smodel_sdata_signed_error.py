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


args = argparse.Namespace()
args.gazeMpiimage_dir = '/project/data/sdata/Image'  # syn data 
args.gazeMpiilabel_dir = '/project/data/sdata/Label'  # syn label
args.output = '/project/results/soutput/snapshots/'  # real model
args.dataset = 'mpiigaze'
args.snapshot='/project/results/soutput/snapshots/'  # real data model
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
snapshot_path = args.snapshot
bins=args.bins
angle=args.angle
bin_width=args.bin_width
cudnn.enabled = True


get_ipython().system('ls ../results')


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


get_ipython().run_cell_magic('time', '', '\npitch_errs = {}\nyaw_errs = {}\n\npitch_xs = []\npitch_ys = []\nyaw_xs = []\nyaw_ys =[]\n\nfor fold in range(15):\n    print(f"fold={fold}")\n    \n    now = datetime.utcnow()\n    now = now.astimezone(timezone(\'US/Pacific\'))\n    date_format=\'%m/%d/%Y %H:%M:%S\'\n    now = now.strftime(date_format)\n    \n    print(args.gazeMpiilabel_dir)\n    folder = os.listdir(args.gazeMpiilabel_dir)\n    folder.sort()  #individual label files\n    testlabelpathcombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder] \n#     print(testlabelpathcombined)\n#     print(args.gazeMpiimage_dir)\n    gaze_dataset=datasets.Mpiigaze(testlabelpathcombined, args.gazeMpiimage_dir, transformations, False, angle, fold)\n\n    test_loader = torch.utils.data.DataLoader(\n        dataset=gaze_dataset,\n        batch_size=batch_size,\n        shuffle=False,\n        num_workers=4,\n        pin_memory=True)\n\n    # list all epochs for testing\n    folder = os.listdir(os.path.join(snapshot_path, "fold" + f\'{fold:0>2}\'))\n    folder.sort(key=natural_keys)\n    folder.pop(-1)  #remove the tensorboard file, now all snapshot files\n#     print(f"folder={folder}")  #contains all the checkpoint files\n                    \n\n    softmax = nn.Softmax(dim=1)\n    configuration = (f"\\ntest configuration equal gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\\n"\n                     f"Start testing dataset={data_set}, FOLD={fold} --{now}---------")\n    print(configuration)\n\n#     outfile.write(configuration)\n    epoch_list=[]\n    avg_MAE=[]\n    for epochs in folder: \n        x = \'\'.join(filter(lambda i: i.isdigit(), epochs))\n        x = int(x)\n        if x != 39:\n            continue\n#         print(f"epochs={epochs}")\n        model=model_used\n        checkpoint = torch.load(os.path.join(snapshot_path+"fold"+f\'{fold:0>2}\', epochs))\n#         print(f"checkpoint={checkpoint}")\n        saved_state_dict = checkpoint[\'model_state_dict\']\n        model= nn.DataParallel(model,device_ids=[0])\n        model.load_state_dict(saved_state_dict)\n        model.cuda(gpu)\n        model.eval()\n        total = 0\n        idx_tensor = [idx for idx in range(28)]\n        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)\n        avg_error = .0\n        \n        print(f"epochs={epochs}")\n        \n        with torch.no_grad():\n            for j, (images, labels, cont_labels, name) in enumerate(test_loader):\n#                 print(f"name={name}")\n                images = Variable(images).cuda(gpu)\n                total += cont_labels.size(0)\n\n                label_pitch = cont_labels[:,0].float()*np.pi/180\n                label_yaw = cont_labels[:,1].float()*np.pi/180\n\n                gaze_pitch, gaze_yaw = model(images)\n\n                # Binned predictions\n                _, pitch_bpred = torch.max(gaze_pitch.data, 1)\n                _, yaw_bpred = torch.max(gaze_yaw.data, 1)\n\n                # Continuous predictions\n                pitch_predicted = softmax(gaze_pitch)\n                yaw_predicted = softmax(gaze_yaw)\n\n                # mapping from binned (0 to 28) to angels (-42 to 42)                \n                pitch_predicted = \\\n                    torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42\n                yaw_predicted = \\\n                    torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42\n\n                pitch_predicted = pitch_predicted*np.pi/180\n                yaw_predicted = yaw_predicted*np.pi/180\n\n#                 print(f"name={name}")\n#                 pitch_errors = []\n#                 yaw_errors = []\n#                 print("label_picthc", len(label_pitch), type(label_pitch), label_pitch)\n#                 print("pitch_predicted", len(pitch_predicted), type(pitch_predicted), pitch_predicted)\n#                 print("names", len(name), type(name), name)\n\n                ## reverse signs and reverse labels\n                label_pitch, label_yaw = label_pitch, label_yaw\n                \n                pitch_errors = (pitch_predicted - label_pitch)*180/np.pi\n                tdict = dict(zip(name, pitch_errors))\n                pitch_errs.update(tdict)\n                \n                yaw_errors = (yaw_predicted - label_yaw)*180/np.pi\n                tdict = dict(zip(name, yaw_errors))\n                yaw_errs.update(tdict)\n                \n                label_pitch = label_pitch*180.0/np.pi\n                label_yaw = label_yaw*180/np.pi\n                pitch_xs.extend(label_pitch.numpy())\n                pitch_ys.extend(pitch_errors.numpy())\n                yaw_xs.extend(label_yaw.numpy())\n                yaw_ys.extend(yaw_errors.numpy())\n                ')


pitch_errs = dict(sorted(pitch_errs.items(), key=lambda item: item[1], reverse=True))
yaw_errs = dict(sorted(yaw_errs.items(), key=lambda item: item[1], reverse=True))


pitch_errors = np.array(list(pitch_errs.values()))
yaw_errors = np.array(list(yaw_errs.values()))





with wandb.init(project="23 best syn model syn data signed errors") as run:
    plt.hist(pitch_errors, bins=40)
    plt.title("signed pitch error distribution: syn model syn data")
    plt.xlabel("degree")
    run.log({"data":wandb.Image(plt)})


with wandb.init(project="23 best syn model syn data signed errors") as run:
    plt.hist(yaw_errors, bins=40)
    plt.title("signed yaw error distribution: syn model syn data")
    plt.xlabel("degree")
    run.log({"data":wandb.Image(plt)})


with wandb.init(project="23 best syn model syn data signed errors") as run:
    data = [[x, y] for (x, y) in zip(pitch_xs, pitch_ys)]
    table = wandb.Table(data=data, columns = ["truth_degree", "error_degree"])
    wandb.log({"pitch_error_scatter" : wandb.plot.scatter(table, "truth_degree", "error_degree", title="pitch error vs ground truth")})


with wandb.init(project="23 best syn model syn data signed errors") as run:
    data = [[x, y] for (x, y) in zip(yaw_xs, yaw_ys)]
    table = wandb.Table(data=data, columns = ["truth_degree", "error_degree"])
    wandb.log({"yaw_error_scatter" : wandb.plot.scatter(table, "truth_degree", "error_degree", title="yaw error vs ground truth")})

