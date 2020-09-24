# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
from dataset import dataset
from PIL import Image
import matplotlib.pyplot as plt
from models.vision import LeNet, weights_init,ResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from utils import label_to_onehot, cross_entropy_for_onehot
import os
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--train', type=bool, default=True, help='leak Train data or Test data')

args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print("Running on %s" % device)

tp = transforms.ToTensor()
tt = transforms.ToPILImage()

net = LeNet().to(device)
net.apply(weights_init)
criterion = cross_entropy_for_onehot
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = dataset(transform=transform,test=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
testset = dataset(transform=transform,test=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

if args.train:
    for i,(image,label) in enumerate(trainloader):
        gt_data = image.to(device)
        gt_onehot_label = label_to_onehot(label.to(device))

        # compute original gradient
        pred = net(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        best_loss = 1e5
        while best_loss > 5:
            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)


            optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

            history = []
            for iters in range(100):
                def closure():
                    optimizer.zero_grad()

                    dummy_pred = net(dummy_data)
                    dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                    dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()

                    return grad_diff

                optimizer.step(closure)
                if iters % 10 == 0:
                    current_loss = closure()
                    if best_loss > current_loss:
                        best_loss = current_loss
                        best_pic = tt(dummy_data[0].cpu())
                        best_label = torch.argmax(dummy_label)
                    print("train",i,iters, "%.4f" % current_loss.item())
                    #history.append(tt(dummy_data[0].cpu()))
        best_pic.save("../.dataset/leakrotationcifar/train/"+str(i) + "_" +str(best_label.item())+".png")

else:
    for i,(image,label) in enumerate(testloader):
        gt_data = image.to(device)
        gt_onehot_label = label_to_onehot(label.to(device))

        # compute original gradient
        pred = net(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        best_loss = 1e5
        while best_loss>5:
            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)


            optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

            history = []
            for iters in range(100):
                def closure():
                    optimizer.zero_grad()

                    dummy_pred = net(dummy_data)
                    dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                    dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()

                    return grad_diff

                optimizer.step(closure)
                if iters % 10 == 0:
                    current_loss = closure()
                    if best_loss > current_loss:
                        best_loss = current_loss
                        best_pic = tt(dummy_data[0].cpu())
                        best_label = torch.argmax(dummy_label)
                    print("test",i,iters, "%.4f" % current_loss.item())

                    #history.append(tt(dummy_data[0].cpu()))
        best_pic.save("../.dataset/leakrotationcifar/test/"+str(i) + "_" +str(best_label.item())+".png")
