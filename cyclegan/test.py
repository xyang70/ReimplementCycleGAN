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

parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', type=int, default=10, help='starting epoch')
parser.add_argument('--dataroot', type=str, default='../datasets/monet2photo/',
                    help='root directory of the dataset')
parser.add_argument('--model', type=str, default='C_GAN.model',
                    help='model file path')
opt = parser.parse_args()
def save_image_internal(image, epoch, batch_id, name, directory):
    file_name = directory+"/" + \
        str(epoch)+"_"+str(batch_id) + "_"+name + ".png"
    image = 0.5*(image.data+1.0)
    save_image(image, file_name)

transform_test = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
testset = ImageDataset(opt.dataroot, transforms_=transform_test, mode='test')
test_loader = DataLoader(testset, batch_size=1, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(opt.model).to(device)
directory = 'test_output'
if not os.path.exists(directory):
    os.makedirs(directory)
model.eval()

for batch_idx, data in enumerate(test_loader):
    A = data['A'].to(device)
    B = data['B'].to(device)
    if A.size() != (1, 3, 256, 256) or B.size() != (1, 3, 256, 256):
        continue
    model.load(A, B)
    # fix these
    fake_B, cyclic_A, fake_A, cyclic_B = model.forward()
    print(batch_idx)
    #dis_A_fake_A,dis_B_fake_B = model.test()
    # loss_B+=lossD_B
    # loss_model_G+=loss_G
    save_image_internal(A, 0, batch_idx, 'input_A', directory)
    save_image_internal(B, 0, batch_idx, 'input_B', directory)
    save_image_internal(cyclic_A, 0, batch_idx, 'cyclic_A', directory)
    save_image_internal(fake_B, 0, batch_idx, 'fake_B', directory)
    save_image_internal(fake_A, 0, batch_idx, 'fake_A', directory)
    save_image_internal(cyclic_B, 0, batch_idx, 'cyclic_B', directory)
# loss_A /= len(testset)
# loss_B /= len(testset)
# loss_model_G /= len(testset)
# print('lossD_A :{},lossD_B:{},loss_G:{}'.format(loss_A, loss_B, loss_model_G))