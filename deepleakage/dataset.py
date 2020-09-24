import torch
from PIL import Image
import os

class dataset(torch.utils.data.Dataset):
    def __init__(self, transform, test=False):
        if not test:
            self.root = "../.dataset/cifar10_rotation_glow/train/"

        else:
            self.root = "../.dataset/cifar10_rotation_glow/test/"

        self.list = []

        for i in range(10):
            paths = os.listdir(self.root+str(i))
            for path in paths:
                self.list.append((self.root+str(i)+"/"+path,i))

        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        (input,label) = self.list[index]
        input = self.transform(Image.open(input))

        return input, label
