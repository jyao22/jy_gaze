import os
import numpy as np
import cv2


import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


class Gaze360(Dataset):
    def __init__(self, path, root, transform, angle, binwidth, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        if train==False:
          angle=90
        self.binwidth=binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)        
        
        # Bin values
        bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])
        

        return img, labels, cont_labels, name

class Mpiigaze(Dataset): 
  def __init__(self, pathorg, root, transform, train, angle, fold=0):
    """ use fold < 0 to denote that the labels are not in folds
    pathorg: a set of paths for the label files
    root: root dir for the iamge data, label file lines contain image file relative to this
    train: True or False """
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    self.n_angle_bins = 0

    lowl = -42  #low limit for angle (for setting bins), -42 in paper
    highl = 42   # high limit for angle (for setting bins), 42 in paper
    bin_size = 3  # 3 degrees in paper

    bins = np.array(range(lowl, highl, bin_size))  #now num_bins is 35
    self.bins = bins
    self.n_angle_bins = len(bins)

    path=pathorg.copy()
    if fold >= 0:
      if train==True:
        path.pop(fold)
      else:
        path=path[fold]
    else:
      if train==True:
        # path.pop(fold)
        trains = [p for p in path if 'train' in p]
        # assert len(trains)==1
        if len(trains)==1:
          path = trains[0]
      else:
        # path=path[fold]
        tests = [p for p in path if 'test' in p]
        # assert len(tests)==1
        if len(tests) == 1:  #otherwise path remains the same
          path = tests[0]
    if isinstance(path, list):
        print(f'path is a list, len(path): {len(path)}')
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      print(f'path={path}')
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                self.lines.append(line)
   
    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))
        
  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[0]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    pitch = label[0]* 180 / np.pi
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)
    
    if self.transform:
        img = self.transform(img)        
    
    # Bin values
    binned_pose = np.digitize([pitch, yaw], self.bins) - 1
    labels = binned_pose
    cont_labels = torch.FloatTensor([pitch, yaw])
    return img, labels, cont_labels, name