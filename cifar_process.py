import torchvision
import torchvision.transforms as transforms

#download cifar
transform = transforms.Compose([
        transforms.ToTensor()
    ])
torchvision.datasets.CIFAR10(root='dataset/cifar10-torchvision', train=True, download=True, transform=transform)
torchvision.datasets.CIFAR10(root='dataset/cifar10-torchvision', train=False, download=True, transform=transform)

#store data by category
import os
files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
from PIL import Image
for i in range(5):
    files[i] = "./dataset/cifar10-torchvision/cifar-10-batches-py/" + files[i]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding="latin1")
    return dict

for file in files:
    dict1 = unpickle(file)
    for i in range(10000):
        pic = dict1['data'][i].reshape(3,32,32).transpose(1,2,0)
        pic = Image.fromarray(pic)
        pic_save_name = "./dataset/cifar10/cifar-10-raw-image/train/"+str(dict1['labels'][i])
        if not os.path.exists(pic_save_name):
            os.makedirs(pic_save_name)
        pic_save_name +="/"+dict1['filenames'][i]
        pic.save(pic_save_name)

test_file = "./dataset/cifar10-torchvision/cifar-10-batches-py/test_batch"
dict1 = unpickle(file)
for i in range(10000):
    pic = dict1['data'][i].reshape(3,32,32).transpose(1,2,0)
    pic = Image.fromarray(pic)
    pic_save_name = "./dataset/cifar10/cifar-10-raw-image/test/"+str(dict1['labels'][i])
    if not os.path.exists(pic_save_name):
        os.makedirs(pic_save_name)
    pic_save_name +="/"+dict1['filenames'][i]
    pic.save(pic_save_name)