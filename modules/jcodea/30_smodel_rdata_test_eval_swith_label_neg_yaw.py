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


wandb.init(project="30_smode_rdata_switch_label_neg_yaw")


# check if we have the correct number of checkpoint files 
ppath ='/project/results/soutput/snapshots/' 
for fold in range(15):
    foldstr = f"fold{fold:0>2}"
    cpath =os.path.join(ppath, foldstr)
    files = os.listdir(cpath)
    print(len(files), end=" ")



args = argparse.Namespace()
args.gazeMpiimage_dir = '/project/data/Image'  #real data 
args.gazeMpiilabel_dir = '/project/data/Label'  #real label
args.output = '/project/results/soutput/snapshots/'
args.dataset = 'mpiigaze'
args.snapshot='/project/results/soutput/snapshots/'
args.evalpath = '/project/results/sroutput/evaluation/'
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

model_used=getArch(arch, bins)  #resnet50 and 28 bins





get_ipython().run_cell_magic('time', '', 'all_MAE = []\n# tosave={}\nfor fold in range(15):\n    print(f"fold={fold}")\n    epoch_values=[]\n    mae_values=[]\n\n\n    now = datetime.utcnow()\n    now = now.astimezone(timezone(\'US/Pacific\'))\n    date_format=\'%m/%d/%Y %H:%M:%S\'\n    now = now.strftime(date_format)\n    \n    print(args.gazeMpiilabel_dir)\n    folder = os.listdir(args.gazeMpiilabel_dir)\n    folder.sort()  #individual label files\n    testlabelpathcombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder] \n    \n    gaze_dataset=datasets.Mpiigaze(testlabelpathcombined, args.gazeMpiimage_dir, transformations, False, angle, fold)\n\n    test_loader = torch.utils.data.DataLoader(\n        dataset=gaze_dataset,\n        batch_size=batch_size,\n        shuffle=True,\n        num_workers=4,\n        pin_memory=True)\n\n    fold_path = os.path.join(evalpath, \'fold\' + f\'{fold:0>2}\'+\'/\')  #for evaluation\n#     print(f"fold_path is {fold_path}")\n    if not os.path.exists(fold_path):\n        os.makedirs(fold_path)\n\n#     if not os.path.exists(os.path.join(evalpath, f"fold"+str(fold))):\n#         os.makedirs(os.path.join(evalpath, f"fold"+str(fold)))\n\n    # list all epoch for testing\n    folder = os.listdir(os.path.join(snapshot_path, "fold" + f\'{fold:0>2}\'))\n    folder.sort(key=natural_keys)\n    folder.pop(-1)  #remove the tensorboard file, now all snapshot files\n#     print(f"folder={folder}")\n                    \n\n    softmax = nn.Softmax(dim=1)\n#     with open(os.path.join(evalpath, os.path.join("fold"+f\'{fold:0>2}\', data_set+".log")), \'w\') as outfile:\n        \n    configuration = (f"\\ntest configuration equal gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\\n"\n                     f"Start testing dataset={data_set}, FOLD={fold} --{now}---------")\n    print(configuration)\n#     tosave[\'time\':now]\n#     outfile.write(configuration)\n    epoch_list=[]\n    avg_MAE=[]\n    for epoch in folder: \n        print(f"entering epoch={epoch}")\n        model=model_used\n        checkpoint = torch.load(os.path.join(snapshot_path+"fold"+f\'{fold:0>2}\', epoch))\n        saved_state_dict = checkpoint[\'model_state_dict\']\n        model= nn.DataParallel(model,device_ids=[0])\n        model.load_state_dict(saved_state_dict)\n        model.cuda(gpu)\n        model.eval()\n        total = 0\n        idx_tensor = [idx for idx in range(28)]\n        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)\n        avg_error = .0\n        with torch.no_grad():\n            for j, (images, labels, cont_labels, name) in enumerate(test_loader):\n                images = Variable(images).cuda(gpu)\n                total += cont_labels.size(0)\n\n                label_pitch = cont_labels[:,0].float()*np.pi/180\n                label_yaw = cont_labels[:,1].float()*np.pi/180\n\n                gaze_pitch, gaze_yaw = model(images)\n\n                # Binned predictions\n                _, pitch_bpred = torch.max(gaze_pitch.data, 1)\n                _, yaw_bpred = torch.max(gaze_yaw.data, 1)\n\n                # Continuous predictions\n                pitch_predicted = softmax(gaze_pitch)\n                yaw_predicted = softmax(gaze_yaw)\n\n                # mapping from binned (0 to 28) to angels (-42 to 42)                \n                pitch_predicted = \\\n                    torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42\n                yaw_predicted = \\\n                    torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42\n\n                pitch_predicted = pitch_predicted*np.pi/180\n                yaw_predicted = yaw_predicted*np.pi/180\n\n                for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):\n                    pl, yl = yl, pl*(-1.0)\n#                     yl = yl*(-1.0)\n                    avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))\n\n        x = \'\'.join(filter(lambda i: i.isdigit(), epoch))\n#             print(f"x={x}")\n        epoch_list.append(x)\n        mean_mae = avg_error/total  #mean mae over the 3000 iamges\n        avg_MAE.append(mean_mae)  \n#             print(f"total={total}")\n        \n        loger = f"[{epoch}---{args.dataset}] Total Num:{total},MAE:{mean_mae}"\n#         outfile.write(loger)\n        print(loger)\n#             print(f"done epoch={epochs}")\n        epochn = int(x)\n#         wandb.log({\'epoch\':epochn,\'total\':total, \'MAE\':mean_mae, \'FOLD\':fold })\n#         tosave ={\'epoch\':epochn,\'total\':total, \'MAE\':mean_mae, \'FOLD\':fold }\n        epoch_values.append(epochn)\n        mae_values.append(mean_mae)\n# #         wandb.log(tosave)\n    all_MAE.append(avg_MAE)\n    data = [[x, y] for (x, y) in zip(epoch_values, mae_values)]\n    table = wandb.Table(data=data, columns = ["x", "y"])\n    wandb.log({"epoch_losses" : wandb.plot.line(table, "x", "y",\n               title="Epoch loss for each fold")})')


all_MAE=np.array(all_MAE)



print(all_MAE.shape)
print(all_MAE.mean(axis=0))
print(all_MAE.mean(axis=1))
print(all_MAE.mean(axis=0).argmin()+1 ,all_MAE.mean(axis=0).min())

