import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

from models import Glow
from tqdm import tqdm

transform_train = transforms.Compose([
    transforms.ToTensor()
])
trainset = torchvision.datasets.CIFAR10(root='dataset/cifar10-torchvision', train=True, download=False, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='dataset/cifar10-torchvision', train=False, download=False, transform=transform_train)
testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

net = Glow(num_channels=512,
           num_levels=3,
           num_steps=16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
net.to(device)
net.load_state_dict({k.replace('module.', ''): v for k, v in
                     torch.load("ckpts/-1.pth.tar")['net'].items()})
net.eval()

q = np.load('rotation.npy')

from PIL import Image

n = 0
for i, (image, label) in enumerate(trainloader):
    print('train,', n)
    z , _ = net(image,reverse=False)
    z = z.view(-1,3*32*32)
    z = z.detach().numpy()
    for i in range(z.shape[0]):
        z[i] = q.dot(z[i])
    z = torch.from_numpy(z).view(-1,3,32,32)
    y , _ = net(z, reverse=True)
    y = torch.sigmoid(y)
    img = Image.fromarray(np.uint8(255*y[0].detach().numpy().transpose(1,2,0)))
    img.save("./dataset/glow-rotation/cifar10_rotation_glow_one_model/train/"+str(label[0].numpy())+"_"+str(n)+'.png')
    n += 1
n = 0
for i, (image, label) in enumerate(testloader):
    print('test,', n)
    z , _ = net(image,reverse=False)
    z = z.view(-1,3*32*32)
    z = z.detach().numpy()
    for i in range(z.shape[0]):
        z[i] = q.dot(z[i])
    z = torch.from_numpy(z).view(-1,3,32,32)
    y , _ = net(z, reverse=True)
    y = torch.sigmoid(y)
    img = Image.fromarray(np.uint8(255*y[0].detach().numpy().transpose(1,2,0)))
    img.save("./dataset/glow-rotation/cifar10_rotation_glow_one_model/test/"+str(label[0].numpy())+"_"+str(n)+'.png')

    n += 1

