import torch
from PIL import Image
import os


class dataset_leak(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.list = []
        self.root = "../.dataset/leakrotationcifar/train/"
        paths = os.listdir(self.root)
        for path in paths:
            self.list.append((self.root + path, int(path.split("_")[1].split(".")[0])))
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        (input, label) = self.list[index]
        input = Image.open(input)
        input = self.transform(input)

        return input, label
