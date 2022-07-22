#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter

import datasets
from model import L2CS
from utils import select_device


get_ipython().system('nvidia-smi')


get_ipython().system('ls /project/data/Label')


get_ipython().system('ls /project/data/Image')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
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
        default= "gaze360", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args





class Nothing:
    pass
args = Nothing()
args.gazeMpiimage_dir = '/project/data/Image'
args.gazeMpiilabel_dir = '/project/data/Label'
args.output = '/project/results/output/snapshots/'
args.dataset = 'mpiigaze'
args.snapshot=''
args.gpu_id = '0,1,2,3'
args.num_epochs = 60
args.batch_size = 100
args.arch = 'ResNet50'
args.alpha = 1.0
args.lr = 0.00001


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
print(gpu)


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





#for tensorboard graphs
folder = os.listdir(args.gazeMpiilabel_dir)
folder.sort()
testlabelpathcombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]
fold = 0

dataset=datasets.Mpiigaze(testlabelpathcombined,args.gazeMpiimage_dir, transformations, True, 180, fold)
train_loader_gaze = DataLoader(
    dataset=dataset,
    batch_size=int(batch_size),
    shuffle=True,
    num_workers=4,
    pin_memory=True)
examples = iter(train_loader_gaze)
example_images,labels_gaze, cont_labels_gaze, name = examples.next()


get_ipython().run_cell_magic('time', '', 'start = time.time()\n\nfolder = os.listdir(args.gazeMpiilabel_dir)\nfolder.sort()\ntestlabelpathcombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]\nfor fold in range(15):\n    model, pre_url = getArch_weights(args.arch, 28)\n    print(fold, model.conv1)\n    load_filtered_state_dict(model, model_zoo.load_url(pre_url))\n#     model = nn.DataParallel(model)\n#     model.to(gpu)\n#     print(\'Again, model.conv1\', model.conv1)\n    print(\'Loading data.\')\n    dataset=datasets.Mpiigaze(testlabelpathcombined,args.gazeMpiimage_dir, transformations, True, 180, fold)\n    \n    train_loader_gaze = DataLoader(\n        dataset=dataset,\n        batch_size=int(batch_size),\n        shuffle=True,\n        num_workers=4,\n        pin_memory=True)\n    \n    torch.backends.cudnn.benchmark = True\n    \n    fold_path = os.path.join(output, \'fold\' + f\'{fold:0>2}\'+\'/\')\n    print(f"fold_path is {fold_path}")\n    if not os.path.exists(fold_path):\n        os.makedirs(fold_path)\n        \n    writer =SummaryWriter(fold_path)\n    example_images = Variable(example_images)\n    writer.add_graph(model, example_images)\n        \n#     summary_name = \'{}_{}\'.format(\'L2CS-mpiigaze\', int(time.time()))\n# #     print(\'summary_name=\', summary_name)\n\n#     if not os.path.exists(os.path.join(output+\'/{}\'.format(summary_name),\'fold\' + str(fold))):\n#         os.makedirs(os.path.join(output+\'/{}\'.format(summary_name),\'fold\' + str(fold)))\n\n    \n    criterion = nn.CrossEntropyLoss().cuda(gpu)\n    reg_criterion = nn.MSELoss().cuda(gpu)\n    softmax = nn.Softmax(dim=1).cuda(gpu)\n    idx_tensor = [idx for idx in range(28)]\n    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)\n\n#     print(\'Again, model.conv1\', model.conv1)\n    # Optimizer gaze\n    #### origianally wrong number of arguments\n    optimizer_gaze = torch.optim.Adam([\n        {\'params\': get_ignored_params(model), \'lr\': 0}, \n        {\'params\': get_non_ignored_params(model), \'lr\': args.lr},\n        {\'params\': get_fc_params(model), \'lr\': args.lr}\n    ], args.lr)\n\n    \n    \n    configuration = f"\\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\\n Start training dataset={data_set}, loader={len(train_loader_gaze)}, fold={fold}--------------\\n"\n#     print(configuration)\n    model.to(gpu)\n    model = nn.DataParallel(model, device_ids=[0,1,2,3])\n    \n    for epoch in range(num_epochs):\n        sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0\n\n        for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):\n            images_gaze = Variable(images_gaze).cuda(gpu)\n\n            # Binned labels\n            label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)\n            label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)\n\n            # Continuous labels\n            label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)\n            label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)\n\n            pitch, yaw = model(images_gaze)\n\n            # Cross entropy loss\n            loss_pitch_gaze = criterion(pitch, label_pitch_gaze)\n            loss_yaw_gaze = criterion(yaw, label_yaw_gaze)\n\n            # MSE loss\n            pitch_predicted = softmax(pitch)\n            yaw_predicted = softmax(yaw)\n\n            # mapping from binned (0 to 28) to angels (-42 to 42) \n            pitch_predicted = \\\n                torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42\n            yaw_predicted = \\\n                torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42\n\n            loss_reg_pitch = reg_criterion(\n                pitch_predicted, label_pitch_cont_gaze)\n            loss_reg_yaw = reg_criterion(\n                yaw_predicted, label_yaw_cont_gaze)\n\n            \n            # Total loss\n            loss_pitch_gaze += alpha * loss_reg_pitch\n            loss_yaw_gaze += alpha * loss_reg_yaw\n\n            sum_loss_pitch_gaze += loss_pitch_gaze\n            sum_loss_yaw_gaze += loss_yaw_gaze\n\n            \n            \n            loss_seq = [loss_pitch_gaze, loss_yaw_gaze]\n            grad_seq = \\\n                [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]\n\n            optimizer_gaze.zero_grad(set_to_none=True)\n            torch.autograd.backward(loss_seq, grad_seq)\n            optimizer_gaze.step()\n\n            iter_gaze += 1\n            yaw_loss = sum_loss_pitch_gaze/iter_gaze\n            pitch_loss = sum_loss_yaw_gaze/iter_gaze\n            \n            \n            iterations = len(dataset)//batch_size\n            div10 = iterations/10\n            if (i+1) % div10 == 0:  #for every div10 batches\n                now=time.time()\n                elapsed = now-start\n                print(f\'Fold: {fold} Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(dataset)//batch_size}] Losses: \'\n                        f\'Gaze Yaw {yaw_loss:.4f},Gaze Pitch {pitch_loss:.3f}\'\n                         f\' elapsed:{elapsed:.1f}s\')\n                \n\n\n        # Save models at numbered epochs.\n#         print("writing to tensor board...")\n        writer.add_scalar(\'pitch_loss\', pitch_loss, epoch)\n        writer.add_scalar(\'yaw_loss\', yaw_loss, epoch)\n        \n        if epoch % 1 == 0 and epoch < num_epochs:\n            print(f"fold_path is still? {fold_path}")\n            pathf = fold_path + \'epoch_\' + str(epoch+1) + \'.pkl\'\n            print(pathf)\n            print(\'Taking snapshot...\')\n            \n            torch.save({\n                \'epoch\': epoch,\n                \'model_state_dict\': model.state_dict(),\n                \'optimizer_state_dict\': optimizer_gaze\n                    .state_dict(),\n                \'pitch_loss\': pitch_loss,\n                \'yaw_loss\': yaw_loss\n                }, pathf)\n            \n    writer.close()   #after each fold\n            \n#             torch.save(model.state_dict(), pathf) \n#                             output+\'/fold\' + str(fold) +\'/\'+\n#                             \'epoch_\' + str(epoch+1) + \'.pkl\')\n#                 )\n\n      ')


get_ipython().system('ls -l /project/results/output')





sys.exit()


get_ipython().system('nvidia-smi')


get_ipython().system('ls')


get_ipython().system('pwd')










