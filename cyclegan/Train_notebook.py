#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use('Agg')
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import datetime
import PIL
import os
import time
import argparse
from CycleGAN import CycleGAN
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse


# ## Optional

# In[2]:
parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', type=int, default=10, help='starting epoch')
parser.add_argument('--dataroot', type=str, default='../datasets/horse2zebra/',
                    help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--size', type=int, default=256,
                    help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3,
                    help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3,
                    help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=8,
                    help='number of cpu threads to use during batch generation')
parser.add_argument('--batchSize', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lambd', type=float, default=10,
                    help='weighr for cycle consistency loss')
parser.add_argument('--lambd_identity', type=float, default=5.,
                    help='weight for identity loss, default 0 means no identity loss')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start_epoch for lr scheduler')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='decay_epoch for lr scheduler')

# In[3]:


opt = parser.parse_args()

# In[4]:


# class opt():
#     batchSize = 4
#     epochs = 100
#     lr = 0.001
#     lambd = 10
#     cuda = True
#     def __init__(self):
#         pass


# ### Hyperparameters

# In[5]:


# hp = opt()
# hp.batchSize


# In[6]:


LR = 0.0002
# batch_size = 200
num_epochs = opt.epochs


# ### Helper Methods

# In[7]:

def plot_graph(num_epochs, acc_list, loss_list):
    #usage : plot_graph(num_epochs,acc_list,loss_list)
    plt.ioff()
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel('Training loss')
    plt.plot(np.arange(num_epochs), loss_list, 'k-')
    plt.title('Training Loss and Training Accuracy')
    plt.xticks(np.arange(num_epochs, dtype=int))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(num_epochs), acc_list, 'b-')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(num_epochs, dtype=int))
    plt.grid(True)
    plt.savefig("plot.png")
    plt.close(fig)


# In[8]:

def save_img_to_para_folder(image, epoch, batch_id, name, directory, opt):
    folder_name = "/ld_" + str(int(opt.lambd)) + "_id_" + str(int(opt.lambd_identity))
    directory = directory + folder_name
    if not os.path.isdir(directory):
        print("Create folder for parameters", directory)
        os.mkdir(directory)
    save_image_internal(image, epoch, batch_id, name, directory)

def save_image_internal(image, epoch, batch_id, name, directory):
    file_name = directory+"/" + \
        str(epoch)+"_"+str(batch_id) + "_"+name + ".png"
    image = 0.5*(image.data+1.0)
    save_image(image, file_name)
    # fig = plt.figure()
    # plt.imshow(image.detach().cpu()[0].permute(1,2,0))
    # fig.savefig(file_name, bbox_inches='tight')
    # plt.close(fig)


# # Main

# ### Load DataSet

# In[9]:


# 256 x 256
transform_train = [transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#transform_test = [transforms.ToTensor(),
#                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
trainset = ImageDataset(
    opt.dataroot, transforms_=transform_train, mode='train')
#testset = ImageDataset(opt.dataroot, transforms_=transform_test, mode='test')
train_loader = DataLoader(trainset, batch_size=opt.batchSize, shuffle=True)
#test_loader = DataLoader(testset, batch_size=opt.batchSize, shuffle=True)
print("Number of training data:", len(trainset))
#print(len(testset))



# ### Declaration

# In[10]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CycleGAN(opt).to(device)

#process path for log, output plot and model
dataset_name = opt.dataroot.split("/")[-2]
directory = "train_output/" + dataset_name
print("DATA SET NAME:", dataset_name)

if not os.path.exists(directory):
    os.makedirs(directory)
log_name = "./log/training_logistics_{}_ld_{}_id_{}.log".format(dataset_name, str(int(opt.lambd)), str(int(opt.lambd_identity)))
open(log_name, 'w').close()


# ### Training

# In[11]:


acc_list = []
loss_list = []

model.train()
for epoch in range(1, num_epochs+1):
    loss_A = 0
    loss_B = 0
    loss_model_G = 0
    start_time = time.time()
    running_loss = 0.0
    acc = 0.0
    print("epoch {}/{}".format(epoch, num_epochs))
    
    for batch_idx, data in enumerate(train_loader):
        A = data['A'].to(device)
        B = data['B'].to(device)


        model.load(A, B)
        lossD_A, lossD_B, loss_G, fake_B, cyclic_A, fake_A, cyclic_B = model.optimize_parameters()
        loss_A += lossD_A.item()
        loss_B += lossD_B.item()
        loss_model_G += loss_G.item()

        if batch_idx % 500 == 0:
            save_img_to_para_folder(A, epoch, batch_idx, 'input_A', directory, opt)
            save_img_to_para_folder(B, epoch, batch_idx, 'input_B', directory, opt)
            save_img_to_para_folder(cyclic_A, epoch, batch_idx,
                                'cyclic_A', directory, opt)
            save_img_to_para_folder(fake_B, epoch, batch_idx, 'fake_B', directory, opt)
            save_img_to_para_folder(fake_A, epoch, batch_idx, 'fake_A', directory, opt)
            save_img_to_para_folder(cyclic_B, epoch, batch_idx,
                                'cyclic_B', directory, opt)
    loss_A /= len(trainset)
    loss_B /= len(trainset)
    loss_model_G /= len(trainset)
    end_time = time.time()
    result = 'TimeStamp:{},Epoch:{}, Training Time: {},lossD_A :{},lossD_B:{},loss_G:{}\n'.format(
        time.ctime(), epoch, end_time-start_time, loss_A, loss_B, loss_model_G)

    model.lr_scheduler_G.step()
    model.lr_scheduler_D.step()

    with open(log_name, "a") as myfile:
        myfile.write(result)
    print(result)
    torch.save(model, "./model/C_GAN_{}_ld_{}_id_{}.model".format(dataset_name, str(int(opt.lambd)), str(int(opt.lambd_identity) )))

print('-' * 20)
