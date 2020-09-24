import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from dataset import dataset
from models import Glow
import util
import torchvision
import os
import tqdm
import random
import torch
from PIL import Image
import os

class imagenet_val(torch.utils.data.Dataset):
    def __init__(self,transform):
        self.transform = transform
        self.paths = []
        root = "~/.imagenet/val/"
        dirs = os.listdir(root)
        #m = 0
        for i in range(len(dirs)):
            tmp = os.listdir(root+dirs[i])
            for p in tmp:
                #m+=1
                #print(m)
                self.paths.append(root+dirs[i]+"/"+p)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):
        flag = True
        while flag:
            try:
                x = Image.open(self.paths[index])
                x = self.transform(x)
                if x.size()[0]!=3:
                    index += 1
                else:
                    flag=False
            except:
                print("error")
                index += 1

        #print(x.size())
        return x , 0


transform = transforms.Compose([
    transforms.Scale((32,32)),
    transforms.ToTensor()
])

for i in range(3):
    net = Glow(num_channels=512,
               num_levels=3,
               num_steps=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    #net.to(device)
    if i==0:
        net.load_state_dict({k.replace('module.', ''): v for k, v in
                             torch.load("ckpts/-2.pth.tar")['net'].items()})
    if i==1:
        net.load_state_dict({k.replace('module.', ''): v for k, v in
                             torch.load("ckpts/-1.pth.tar")['net'].items()})
    net.eval()
    #testset = dataset(-2, transform, test=True,rotation_data=True)
    testset = torchvision.datasets.CIFAR10(root='dataset/cifar10-torchvision', train=False, download=True, transform=transform)
    #testset = imagenet_val(transform)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
    loss_fn = util.NLLLoss().to(device)
    loss_meter = util.AverageMeter()
    bpd_sum = 0
    n = 0
    for x, _ in testloader:
        #x = x.to(device)
        z, sldj = net(x, reverse=False)
        loss = loss_fn(z, sldj)
        loss_meter.update(loss.item(), x.size(0))
        n+=1
        bpd_sum += util.bits_per_dim(x, loss_meter.avg)
        #print(util.bits_per_dim(x, loss_meter.avg))
        #print(bpd_sum/n)
    print(bpd_sum/n)

for i in range(3):
    net = Glow(num_channels=512,
               num_levels=3,
               num_steps=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    #net.to(device)
    if i==0:
        net.load_state_dict({k.replace('module.', ''): v for k, v in
                             torch.load("ckpts/-2.pth.tar")['net'].items()})
    if i==1:
        net.load_state_dict({k.replace('module.', ''): v for k, v in
                             torch.load("ckpts/-1.pth.tar")['net'].items()})
    net.eval()
    testset = imagenet_val(transform)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
    loss_fn = util.NLLLoss().to(device)
    loss_meter = util.AverageMeter()
    bpd_sum = 0
    n = 0
    for x, _ in testloader:
        #x = x.to(device)
        z, sldj = net(x, reverse=False)
        loss = loss_fn(z, sldj)
        loss_meter.update(loss.item(), x.size(0))
        n+=1
        bpd_sum += util.bits_per_dim(x, loss_meter.avg)
        #print(util.bits_per_dim(x, loss_meter.avg))
        #print(bpd_sum/n)
    print(bpd_sum/n)
