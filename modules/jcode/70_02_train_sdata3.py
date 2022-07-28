#!/usr/bin/env python
# coding: utf-8

# training script for training sdata3


import os
import sys
import argparse
import time
from datetime import datetime
from pytz import timezone
import numpy as np

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import wandb

import datasets
from model import L2CS
from utils import select_device


sys.path.append('/project/modules/jmodules')
from jutils import SynJSON as SJ, get_now


# !nvidia-smi


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
        default='/project/data/sdata3/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='/project/data/sdata3/Label', type=str)

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
        default='0,1,2,3', type=str)
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

wandbproject = 'sdata3_training'


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
                
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def getArch_weights(arch, bins):
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url


# args = parse_args()
cudnn.enabled = True
num_epochs = args.num_epochs
batch_size = args.batch_size

gpu = select_device(args.gpu_id, batch_size=args.batch_size)
# print(gpu)

data_set=args.dataset
alpha = args.alpha
output=args.output
transformations = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


start = time.time()
folder = os.listdir(args.gazeMpiilabel_dir)
folder.sort()
testlabelpathcombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]
dataset = datasets.Mpiigaze(testlabelpathcombined,args.gazeMpiimage_dir, transformations, True, 180, fold=-1)
num_bins = dataset.n_angle_bins

wandb.init(project=wandbproject, name='sdata3_train_errors')
print(f'num_bins={num_bins}')
print(f'total lines = {len(dataset.lines)}')
model, pre_url = getArch_weights(args.arch, num_bins)
print(model.conv1)
load_filtered_state_dict(model, model_zoo.load_url(pre_url))
print('Loading data.')
# dataset=datasets.Mpiigaze(testlabelpathcombined,args.gazeMpiimage_dir, transformations, True, 180, fold)

train_loader_gaze = DataLoader(
    dataset=dataset,
    batch_size=int(batch_size),
    shuffle=True,
    num_workers=8,
    pin_memory=True)

torch.backends.cudnn.benchmark = True
# fold_path = os.path.join(output, 'fold' + f'{fold:0>2}'+'/')
now=get_now()
print(f"output is {output} {now}")
if not os.path.exists(output):
     os.makedirs(output)

criterion = nn.CrossEntropyLoss().cuda(gpu)
reg_criterion = nn.MSELoss().cuda(gpu)
softmax = nn.Softmax(dim=1).cuda(gpu)
idx_tensor = [idx for idx in range(num_bins)]
idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

#### origianally wrong number of arguments
optimizer_gaze = torch.optim.Adam([
    {'params': get_ignored_params(model), 'lr': 0}, 
    {'params': get_non_ignored_params(model), 'lr': args.lr},
    {'params': get_fc_params(model), 'lr': args.lr}
], args.lr)



configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}--------------\n"
print(configuration)
model.to(gpu)
model = nn.DataParallel(model, device_ids=[0,1,2,3])
now = get_now()
print(f'start training at {now}')
for epoch in range(num_epochs):
    sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

    for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
        images_gaze = Variable(images_gaze).cuda(gpu)
#         print(f'epoch : {epoch}, batch: {i}', end= ' ')
        # Binned labels
        label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
        label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

        # Continuous labels
        label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
        label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

        pitch, yaw = model(images_gaze)

        # Cross entropy loss
        loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
        loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

        # MSE loss
        pitch_predicted = softmax(pitch)
        yaw_predicted = softmax(yaw)

        # mapping from binned (0 to 28) to angels (-52 to 52) 
        pitch_predicted =             torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 52
        yaw_predicted =             torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 52

        loss_reg_pitch = reg_criterion(
            pitch_predicted, label_pitch_cont_gaze)
        loss_reg_yaw = reg_criterion(
            yaw_predicted, label_yaw_cont_gaze)

        # Total loss
        loss_pitch_gaze += alpha * loss_reg_pitch
        loss_yaw_gaze += alpha * loss_reg_yaw

        sum_loss_pitch_gaze += loss_pitch_gaze
        sum_loss_yaw_gaze += loss_yaw_gaze

        loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
        grad_seq =             [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

        optimizer_gaze.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer_gaze.step()

        iter_gaze += 1
        yaw_loss = sum_loss_pitch_gaze/iter_gaze
        pitch_loss = sum_loss_yaw_gaze/iter_gaze

        iterations = len(dataset)//batch_size
        div10 = iterations/10
        if (i) % div10 == 0:  #for every div10 batches
            now=time.time()
            elapsed = now-start
            print(f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(dataset)//batch_size}] Losses: '
                    f'Gaze Yaw {yaw_loss:.4f},Gaze Pitch {pitch_loss:.3f}'
                     f' elapsed:{elapsed:.1f}s')

    wandb.log({f'pitch_loss':pitch_loss, f'yaw_loss':yaw_loss }, step=epoch+1)
    
    if epoch % 1 == 0 and epoch < num_epochs:
        now=get_now()
        print(f"epoch = {epoch+1}, {now}")
        checkf = 'epoch_' + str(epoch+1) + '.pkl'
        pathf = os.path.join(output,checkf)
        print(pathf)
        print('Taking snapshot...')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_gaze
                .state_dict(),
            'pitch_loss': pitch_loss,
            'yaw_loss': yaw_loss
            }, pathf)

wandb.finish()
















