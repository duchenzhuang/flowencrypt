import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from dataset import dataset
from models import Glow
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor()
])


for num_class in range(10):
    trainset = dataset(num_class % 10, transform, test=False)
    trainloader = data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=8)

    testset = dataset(num_class % 10, transform, test=True)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

    net = Glow(num_channels=512,
               num_levels=3,
               num_steps=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    net.to(device)
    net.load_state_dict({k.replace('module.', ''): v for k, v in
                         torch.load("ckpts/"+str(num_class)+".pth.tar")['net'].items()})
    net.eval()
    q = np.load('rotation.npy')



    n = 0
    for i, (image, label) in enumerate(trainloader):
        print('num_class:{},train:{},'.format(num_class,n))

        z , _ = net(image,reverse=False)
        z = z.view(-1,3*32*32)
        z = z.detach().numpy()
        for i in range(z.shape[0]):
            z[i] = q.dot(z[i])
        z = torch.from_numpy(z).view(-1,3,32,32)
        y , _ = net(z, reverse=True)
        y = torch.sigmoid(y)
        img = Image.fromarray(np.uint8(255*y[0].detach().numpy().transpose(1,2,0)))
        img.save("./dataset/glow-rotation/cifar10_rotation_glow/train/"+str(label[0].numpy())+"/"+str(n)+'.png')
        n += 1

    n = 0
    for i, (image, label) in enumerate(testloader):
        print('num_class:{},test:{},'.format(num_class,n))

        z , _ = net(image,reverse=False)
        z = z.view(-1,3*32*32)
        z = z.detach().numpy()
        for i in range(z.shape[0]):
            z[i] = q.dot(z[i])
        z = torch.from_numpy(z).view(-1,3,32,32)
        y , _ = net(z, reverse=True)
        y = torch.sigmoid(y)
        img = Image.fromarray(np.uint8(255*y[0].detach().numpy().transpose(1,2,0)))
        img.save("./dataset/glow-rotation/cifar10_rotation_glow/test/"+str(label[0].numpy())+"/"+str(n)+'.png')
        n += 1

