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


get_ipython().run_cell_magic('time', '', 'all_MAE = []\n# tosave={}\nfor fold in range(15):\n#     frun = wandb.init(project="51_syn_mode_real_data")\n#     frun.name = f\'fold{fold}_mae\'\n    print(f"fold={fold}")\n#     epoch_values=[]\n#     mae_values=[]\n    now = get_now()\n#     print(args.gazeMpiilabel_dir)\n#     folder = os.listdir(args.gazeMpiilabel_dir)\n#     folder.sort()  #individual label files\n#     testlabelpathcombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder] \n    \n#     gaze_dataset=datasets.Mpiigaze(testlabelpathcombined, args.gazeMpiimage_dir, transformations, False, angle, fold)\n\n#     test_loader = torch.utils.data.DataLoader(\n#         dataset=gaze_dataset,\n#         batch_size=batch_size,\n#         shuffle=True,\n#         num_workers=4,\n#         pin_memory=True)\n\n#     fold_path = os.path.join(evalpath, \'fold\' + f\'{fold:0>2}\'+\'/\')  #for evaluation\n#     print(f"fold_path is {fold_path}")\n#     if not os.path.exists(fold_path):\n#         os.makedirs(fold_path)\n\n#     if not os.path.exists(os.path.join(evalpath, f"fold"+str(fold))):\n#         os.makedirs(os.path.join(evalpath, f"fold"+str(fold)))\n\n    # list all epoch for testing\n    folder = os.listdir(os.path.join(snapshot_path, "fold" + f\'{fold:0>2}\'))\n    print(folder)\n    folder.sort(key=natural_keys)\n#     folder.pop(-1)  #remove the tensorboard file, now all snapshot files\n    print(f"folder={folder}")\n  ')




    softmax = nn.Softmax(dim=1)
#     with open(os.path.join(evalpath, os.path.join("fold"+f'{fold:0>2}', data_set+".log")), 'w') as outfile:
        
    configuration = (f"\ntest configuration equal gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\n"
   f"Start testing dataset={data_set}, FOLD={fold} --{now}---------")
    print(configuration)
#     tosave['time':now]
#     outfile.write(configuration)
    epoch_list=[]
    avg_MAE=[]
    for epoch in folder: 
        print(f"entering epoch={epoch}")
        model=model_used
        checkpoint = torch.load(os.path.join(snapshot_path+"fold"+f'{fold:0>2}', epoch))
        saved_state_dict = checkpoint['model_state_dict']
        model= nn.DataParallel(model,device_ids=[0])
        model.load_state_dict(saved_state_dict)
        model.cuda(gpu)
        model.eval()
        total = 0
        idx_tensor = [idx for idx in range(35)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
        avg_error = .0
        with torch.no_grad():
            for j, (images, labels, cont_labels, name) in enumerate(test_loader):
                images = Variable(images).cuda(gpu)
                total += cont_labels.size(0)

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
                pitch_predicted =   torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 52
                yaw_predicted =   torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 52

                pitch_predicted = pitch_predicted*np.pi/180
                yaw_predicted = yaw_predicted*np.pi/180

                for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
  pl, yl = yl, pl*(-1.0)
#                     yl = yl*(-1.0)
  avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))

        x = ''.join(filter(lambda i: i.isdigit(), epoch))
#             print(f"x={x}")
        
        epoch_list.append(x)
        mean_mae = avg_error/total  #mean mae over the 3000 iamges
        avg_MAE.append(mean_mae)  
#             print(f"total={total}")
        now = get_now()
        loger = f"[{epoch}---{args.dataset}] Total Num:{total},MAE:{mean_mae}  {now}"
#         outfile.write(loger)
        print(loger)
#             print(f"done epoch={epochs}")
        epochn = int(x)
#         wandb.log({'epoch':epochn,'total':total, 'MAE':mean_mae, 'FOLD':fold })
#         tosave ={'epoch':epochn,'total':total, 'MAE':mean_mae, 'FOLD':fold }
        epoch_values.append(epochn)
        mae_values.append(mean_mae)
# #         wandb.log(tosave)
        wandb.log({'epoch': epoch, f'try3_fold_{fold}_avg_mae':mean_mae}, step=epochn)
       
    all_MAE.append(avg_MAE)
    wandb.finish()


all_MAE=np.array(all_MAE)



print(all_MAE.shape)
print(all_MAE.mean(axis=0))
print(all_MAE.mean(axis=1))
print(all_MAE.mean(axis=0).argmin()+1 ,all_MAE.mean(axis=0).min())


best_epoch = all_MAE.mean(axis=0).argmin()+1
min_error = all_MAE.mean(axis=0).min()
mean_errors = all_MAE.mean(axis=0)
print(best_epoch,min_error, mean_errors.shape )


xx = mean_errors
min_error = 100
with wandb.init(project="51_syn_mode_real_data") as run:
    run.name = "mean_error_per_epoch"
    for epoch in range(len(xx)):
        err = xx[epoch]
        if err < min_error:
            min_error = err
            run.summary["minimum_error (degree)"] = min_error
        run.log({f'mean epoch error (degree)':err}, step=epoch)
        


xx = 10*np.random.random((5, 6))
for row in xx:
    print(row)
    print(row.min(), np.argmin(row)+1)


for fdata in all_MAE:
    best_epoch = np.argmin(fdata)+1
    min_at_best
    

