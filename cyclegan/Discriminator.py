import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc = 64, layer_n = 3, mul_nc = 2):
        super(Discriminator, self).__init__()
        """
        input_nc: int, number of channel for input of the first layer
        output_nc: int, number of channel for output of the first layer
        layer_n: int, number of the baisc layer for the discriminator
        mul_nc: int, the factor of channel number between previous layer and current layer
        """
        #define the first layer
        self.output_nc = output_nc
        model = [nn.Conv2d(input_nc, self.output_nc, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True) ]
        #add basic layer
        for i in range(layer_n):
            input_nc = self.output_nc 
            self.output_nc *= mul_nc
            model += [nn.Conv2d(input_nc, self.output_nc, 4, stride=2, padding=1),
                      nn.InstanceNorm2d(self.output_nc),
                      nn.LeakyReLU(0.2, inplace=True) 
        # FCN classification layer
        # Reduce to only 1 channel
        model += [nn.Conv2d(self.output_nc, 1, 4, padding=1),     #output_nc = 512
                  nn.InstanceNorm2d(1),
                  nn.LeakyReLU(0.2, inplace=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        input 
            x: dimension = (size of mini-batch, channel, w, h)
        output: 
            output: dimension = (size of mini-batch, 1), score for each image
        """
        x =  self.model(x)
        # Average pooling and flatten
        # Output 1-D score by using the kernal size same as the final image 
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
    def set_grad(self, requires_grad=False):
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            requires_grad: bool, whether the networks require gradients or not
        """
        for param in self.model.parameters():
            param.requires_grad = requires_grad
