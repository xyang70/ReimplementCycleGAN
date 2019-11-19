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
parser.add_argument('--dataroot', type=str, default='../datasets/monet2photo/',
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
parser.add_argument('--epochs', type=int, default=75, help='number of epochs')
parser.add_argument('--lambd', type=float, default=10,
                    help='weighr for cycle consistency loss')
parser.add_argument('--lambd_identity', type=float, default=0.,
                    help='weight for identity loss, default 0 means no identity loss')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start_epoch for lr scheduler')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='decay_epoch for lr scheduler')

# In[3]:


#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batchSize', type=int, default=1, help='batch size')
#     parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
#     parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
#     parser.add_argument('--lambda', type=float, default=10, help='weighr for cycle consistency loss')
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
transform_test = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
trainset = ImageDataset(
    opt.dataroot, transforms_=transform_train, mode='train')
testset = ImageDataset(opt.dataroot, transforms_=transform_train, mode='test')
train_loader = DataLoader(trainset, batch_size=opt.batchSize, shuffle=True)
test_loader = DataLoader(testset, batch_size=opt.batchSize, shuffle=True)
print(len(trainset))
print(len(testset))


# ### Declaration

# In[10]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CycleGAN(opt).to(device)
directory = 'train_output'
if not os.path.exists(directory):
    os.makedirs(directory)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)


# ### Training

# In[11]:


acc_list = []
loss_list = []
loss_A = 0
loss_B = 0
loss_model_G = 0
model.train()
for epoch in range(1, num_epochs+1):

    start_time = time.time()
    running_loss = 0.0
    acc = 0.0
    print("epoch {}/{}".format(epoch, num_epochs))
    for batch_idx, data in enumerate(train_loader):
        # print('--------')
        # print(data)
        A = data['A'].to(device)
        # print(A.size())
        B = data['B'].to(device)
        if A.size() != (1, 3, 256, 256) or B.size() != (1, 3, 256, 256):
            continue
        # print(B.shape)
        # show_image(A)
        # show_image(B)
        model.load(A, B)
        lossD_A, lossD_B, loss_G, fake_B, cyclic_A, fake_A, cyclic_B = model.optimize_parameters()
        loss_A += lossD_A.item()
        loss_B += lossD_B.item()
        loss_model_G += loss_G.item()
        if batch_idx % 2000 == 0:
            save_image_internal(A, epoch, batch_idx, 'input_A', directory)
            save_image_internal(B, epoch, batch_idx, 'input_B', directory)
            save_image_internal(cyclic_A, epoch, batch_idx,
                                'cyclic_A', directory)
            save_image_internal(fake_B, epoch, batch_idx, 'fake_B', directory)
            save_image_internal(fake_A, epoch, batch_idx, 'fake_A', directory)
            save_image_internal(cyclic_B, epoch, batch_idx,
                                'cyclic_B', directory)
#         break
    loss_A /= len(trainset)
    loss_B /= len(trainset)
    loss_model_G /= len(trainset)
    end_time = time.time()
    result = 'TimeStamp:{},Epoch:{}, Training Time: {},lossD_A :{},lossD_B:{},loss_G:{}\n'.format(
        time.ctime(), epoch, end_time-start_time, loss_A, loss_B, loss_model_G)
    with open("training_logistics.txt", "a") as myfile:
        myfile.write(result)
    print(result)
    torch.save(model, 'C_GAN.model')
#     correct = 0
#     with torch.no_grad():
#         #model.eval()
#         start_time = time.time()
#         for batch_idx, (A,B) in enumerate(test_loader):
#             A = A.to(device)
#             B = B.to(device)
#             model.load(A,B)
#             #model.optimize_parameters()
#             #_,predicted = torch.max(outputs.data,1)
#             #correct+=torch.sum(predicted==labels).item()
#         end_time = time.time()
#         print('Testing Time: ',end_time-start_time ,'s, Testing Accurarcy: ',correct/len(testset))
print('-' * 20)


# # In[ ]:
# directory = 'test_output'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# model.eval()
# loss_A = 0
# loss_B = 0
# loss_model_G = 0
# epoch = 0
# for batch_idx, data in enumerate(test_loader):
#     A = data['A'].to(device)
#     B = data['B'].to(device)
#     if A.size() != (1, 3, 256, 256) or B.size() != (1, 3, 256, 256):
#         continue
#     model.load(A, B)
#     # fix these
#     fake_B, cyclic_A, fake_A, cyclic_B = model.forward()
#     #dis_A_fake_A,dis_B_fake_B = model.test()
#     # loss_B+=lossD_B
#     # loss_model_G+=loss_G
#     save_image_internal(A, epoch, batch_idx, 'input_A', directory)
#     save_image_internal(B, epoch, batch_idx, 'input_B', directory)
#     save_image_internal(cyclic_A, epoch, batch_idx, 'cyclic_A', directory)
#     save_image_internal(fake_B, epoch, batch_idx, 'fake_B', directory)
#     save_image_internal(fake_A, epoch, batch_idx, 'fake_A', directory)
#     save_image_internal(cyclic_B, epoch, batch_idx, 'cyclic_B', directory)
# loss_A /= len(testset)
# loss_B /= len(testset)
# loss_model_G /= len(testset)
# print('lossD_A :{},lossD_B:{},loss_G:{}'.format(loss_A, loss_B, loss_model_G))


# In[ ]:


# plot_graph(num_epochs,acc_list,loss_list)
# plot_graph(num_epochs,acc_list,loss_list)


# In[ ]:


# torch.save(model,'cycleGAN.model')
