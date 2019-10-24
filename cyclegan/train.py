import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import datetime
import PIL
import os
import argparse
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset

from CycleGAN import CycleGAN


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--lambda', type=float, default=10, help='weighr for cycle consistency loss')

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #model = CycleGAN(opt)
    transforms_ = [transforms.RandomHorizontalFlip(),
                   transforms.ToTensor()]

    
    dataloader = DataLoader(ImageDataset('../datasets/monet2photo/', transforms_=transforms_), 
                        batch_size=opt.batchSize, shuffle=True)


    """
    # within each epoch

        # time each iter

        # for each image "pair"
        model.load(img_A, imgB)
        model.optimize_parameters()

        # save training stats to file

        # plot graph if possible
    """




