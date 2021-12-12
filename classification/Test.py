import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
from ClassficationNetwork import CNN

data_dir = "/home/duhuaiyu/Downloads/facemaskdata/classification_data"
train_dir = data_dir + '/train'
valid_dir = data_dir + '/validation'
filename= 'checkpoint24_acc90.pth'


data_transforms = {
    'train':
        transforms.Compose([
        # transforms.RandomRotation(30),#随机旋转，-45到45度之间随机选
        # # transforms.CenterCrop(224),#从中心开始裁剪
        # transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        # transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        # transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        # transforms.Normalize(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 100
#image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
image_dataset = datasets.ImageFolder('/home/duhuaiyu/Downloads/facemaskdata/images/', data_transforms['test'])
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

class_names = image_dataset.classes
print(dataloader)

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = CNN()
checkpoint = torch.load('checkpoint24_acc90.pth')
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

model_ft = model_ft.to(device)
model_ft.eval()
# inputs = dataloaders['test'].to(device)

if train_on_gpu:
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print('Test Acc: {:.4f}'.format(epoch_acc))
else:
    output = model_ft(dataloader)