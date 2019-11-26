import torch.nn as nn

BATCH_SIZE = 1
# IMG_WIDTH = 256
# IMG_HEIGHT = 256
IMG_CHANNEL = 3
BASE_GEN_FEATURE = 64

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(BASE_GEN_FEATURE*4, BASE_GEN_FEATURE*4, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(BASE_GEN_FEATURE*4)
        )
        self.relu = nn.Sequential(
            nn.ReLU()
        )

    def forward(self, x):
        fx = self.block(x)
        fx = self.relu(fx)
        fx = self.block(fx)
        return x+fx

class Generator(nn.Module):
    def __init__(self, n_blocks=9):
        super(Generator, self).__init__() #256x256x3
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3), #260x260x3
            nn.Conv2d(IMG_CHANNEL, BASE_GEN_FEATURE, kernel_size=7, stride=1, padding=0), #256x256x32
            nn.InstanceNorm2d(BASE_GEN_FEATURE*BATCH_SIZE),
            nn.ReLU(inplace=True),
            nn.Conv2d(BASE_GEN_FEATURE, BASE_GEN_FEATURE*2, kernel_size=3, stride=2, padding=1), #128x128x64
            nn.InstanceNorm2d(BASE_GEN_FEATURE*2*BATCH_SIZE),
            nn.ReLU(inplace=True),
            nn.Conv2d(BASE_GEN_FEATURE*2, BASE_GEN_FEATURE*4, kernel_size=3, stride=2, padding=1), #64x64x128
            nn.InstanceNorm2d(BASE_GEN_FEATURE*4*BATCH_SIZE),
            nn.ReLU(inplace=True)
        )
        self.n_blocks = n_blocks
        self.resblock = ResnetBlock()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(BASE_GEN_FEATURE*4, BASE_GEN_FEATURE*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(BASE_GEN_FEATURE*2*BATCH_SIZE),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(BASE_GEN_FEATURE*2, BASE_GEN_FEATURE, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(BASE_GEN_FEATURE*BATCH_SIZE),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(BASE_GEN_FEATURE, IMG_CHANNEL, kernel_size=7, stride=1),
            #nn.InstanceNorm2d(IMG_CHANNEL*BATCH_SIZE),
            # nn.ReLU(inplace=True),
            nn.Tanh()
        )


    def forward(self, x): #256x256x3
        x = self.encoder(x)
        for i in range(self.n_blocks):
            x = self.resblock(x)
        x = self.decoder(x)
        return x
