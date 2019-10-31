import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import random 

from Generator import Generator
from Discriminator import Discriminator

class ReplayBuffer(object):
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.size = 0

    def add_and_sample(self, img, batch_size):
        if random.random() > 0.5 or self.size < 1:
            ret = img
        else:
            ret = random.sample(self.buffer, batch_size)

        if size < capacity:
            self.buffer.append(img)
        else:
            self.buffer[self.position] = img
        self.position = (self.position + 1) % self.capacity
        self.size += 1

        return ret

class CycleGAN(nn.Module):
    def __init__(self, opt, is_train=True):
        super(CycleGAN, self).__init__()
        self.is_train = is_train

        self.genA2B = Generator()
        self.genB2A = Generator()
        self.disA = Discriminator()
        self.disB = Discriminator()

        self.criterionGAN = nn.MSELoss()
        self.criterionCycle = nn.L1Loss()

        self.lr_scheduler = None
        self.optimizer_G = optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=opt.lr)
        self.optimizer_D = optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=opt.lr)
        self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.fake_As = ReplayBuffer()
        self.fake_Bs = ReplayBuffer()

    def load(self, A, B):
        self.real_A = A
        self.real_B = B

    def forward(self, x): 
        self.fake_B = self.genA2B(self.real_A)
        self.cyclic_A = self.genB2A(self.fake_B)
        self.fake_A = self.genB2A(self.real_B)
        self.cyclic_A = self.genA2B(self.fake_A)

    def backward_D(self, D, real, fake):
        D_real = D(real)
        loss_D_real = self.criterionGAN(D_real, True)

        D_fake = D(fake.detach())
        loss_D_fake = self.criterionGAN(D_fake, False)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        return loss_D

    def optimize_parameters(self):
        self.forward()

        # optimize Generator: calc loss of G -> backward -> update weights
        self.disA.set_grad(False)
        self.disB.set_grad(False)

        self.optimizer_G.zero_grad()
        self.loss_genA2B = self.criterionGAN(self.disA(self.fake_B), True)
        self.loss_cyclic_A = self.criterionCycle(self.cyclic_A, self.real_A)
        self.loss_genB2A = self.criterionGAN(self.disB(self.fake_A), True)
        self.loss_cyclic_B = self.criterionCycle(self.cyclic_B, self.real_B)
        self.loss_G = loss_genA2B + loss_genB2A + opt.lambd * (loss_cyclic_A + loss_cyclic_B)

        self.loss_G.backward()

        self.optimizer_G.step()
        
        # optimize Discriminator: calc loss of D -> backward -> update weights
        self.disA.set_grad(True)
        self.disB.set_grad(True)

        self.optimizer_D.zero_grad()

        fake_A = self.fake_As.add_and_sample(self.fake_A)
        loss_D_A = self.backward_D(self.disA, self.real_A, fake_A)
        fake_B = self.fake_Bs.add_and_sample(self.fake_B)
        loss_D_B = self.backward_D(self.disB, self.real_B, fake_B)

        self.optimizer_D.step()
        return loss_D_A, loss_D_B, loss_G
