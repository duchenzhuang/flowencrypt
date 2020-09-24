Libraries are required:
    Python >= 3.6
    PyTorch >= 1.0
    torchvision >= 0.4

Notice that we take CIFAR10 as the training dataset. If you want to train glows on lsun, you just need to modify the dataloader.
You need to store per category data in one folder for easy to load and the checkpoint is named by the "num_class".
You can view the structure of the data store in "dataset" folder.

You should run cifar_process.py to download and process cifar before doing the following.

Supervised Learning:
    #train glows on cifar-10, and num_class represents which class's data you use.
    python train.py --num_class=0
    
    #rotate data, and the data is saved in "dataset/glow-rotation/cifar10_rotation_glow"
    python gen_rotation_matrix.py
    python rotate_data.py
    
    #classifiers
    python classifier/main.py
    
    
Unsupervised Learning:
    #Train one glow on cifar10
    python train.py --num_class=-1
    
    #rotate data use the above glow, and the data is saved in "dataset/glow-rotation/cifar10_rotation_glow_one_model"
    python rotate_data_unsuper.py
    
    #Train the second glow using the encrypted data
    python train.py --num_class=-2
    
    #calculate bpd
    python test_bpd.py
    
    #Some visualization results can be check in two-glows.ipynb
   
Deep Leakage:
    #leak data from gradients
    python deepleakage/main.py --train=True
    
    #Use leaked data to train classifiers
    python classifier/main.py --leaked=True
     
    