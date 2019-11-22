import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import random
import itertools
from Generator import Generator
from Discriminator import Discriminator

Tensor = torch.cuda.FloatTensor
#if opt.cuda else torch.Tensor
#Tensor = torch.Tensor

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0),"Decay must start before the training sessions ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


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
            ret = torch.cat(ret, 0)

        if self.size < self.capacity:
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
        self.opt = opt
        self.genA2B = Generator()
        self.genB2A = Generator()
        self.disA = Discriminator()
        self.disB = Discriminator()
        self.genA2B.apply(self.weights_init_normal)
        self.genB2A.apply(self.weights_init_normal)
        self.disA.apply(self.weights_init_normal)
        self.disB.apply(self.weights_init_normal)

        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionCycle = nn.L1Loss()
        if self.opt.lambd_identity > 0:
            self.criterionIdentity = nn.L1Loss()

        self.optimizer_G = optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.opt.lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.opt.lr, betas=(0.5, 0.999))
        # self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(self.opt.epochs, self.opt.start_epoch, self.opt.decay_epoch).step)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=LambdaLR(self.opt.epochs, self.opt.start_epoch, self.opt.decay_epoch).step)

        self.fake_As = ReplayBuffer()
        self.fake_Bs = ReplayBuffer()

    def load(self, A, B):
        self.real_A = A
        self.real_B = B

    def weights_init_normal(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self):
        self.fake_B = self.genA2B(self.real_A)
        self.cyclic_A = self.genB2A(self.fake_B)
        self.fake_A = self.genB2A(self.real_B)
        self.cyclic_B = self.genA2B(self.fake_A)
        return self.fake_B, self.cyclic_A, self.fake_A, self.cyclic_B

    def backward_D(self, D, real, fake):
        D_real = D(real)[0]
        loss_D_real = self.criterionGAN(D_real, Tensor(1).fill_(1.0))

        D_fake = D(fake.detach())[0]
        loss_D_fake = self.criterionGAN(D_fake, Tensor(1).fill_(0.0))

        loss_D = (loss_D_real + loss_D_fake)*0.5
        loss_D.backward()
        return loss_D

    def optimize_parameters(self):
        self.forward()

        # optimize Generator: calc loss of G -> backward -> update weights
        self.disA.set_grad(False)
        self.disB.set_grad(False)

        self.optimizer_G.zero_grad()

        if self.opt.lambd_identity > 0:
            identity_A = self.genB2A(self.real_A)
            self.loss_identity_A = self.criterionIdentity(self.real_A, identity_A) * self.opt.lambd_identity
            identity_B = self.genA2B(self.real_B)
            self.loss_identity_B = self.criterionIdentity(self.real_B, identity_B) * self.opt.lambd_identity
            

        self.loss_genA2B = self.criterionGAN(self.disA(self.fake_B)[0], Tensor(1).fill_(1.0))
        self.loss_cyclic_A = self.criterionCycle(self.cyclic_A, self.real_A)
        self.loss_genB2A = self.criterionGAN(self.disB(self.fake_A)[0], Tensor(1).fill_(1.0))
        self.loss_cyclic_B = self.criterionCycle(self.cyclic_B, self.real_B)
        self.loss_G = self.loss_genA2B + self.loss_genB2A + self.opt.lambd * (self.loss_cyclic_A + self.loss_cyclic_B)  #opt.lambd

        if self.opt.lambd_identity > 0:
            self.loss_G += self.loss_identity_A + self.loss_identity_B

        self.loss_G.backward()

        for group in self.optimizer_G.param_groups:
            for p in group['params']:
                state = self.optimizer_G.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1022):
                        state['step'] = 1000

        self.optimizer_G.step()

        # optimize Discriminator: calc loss of D -> backward -> update weights
        self.disA.set_grad(True)
        self.disB.set_grad(True)

        self.optimizer_D.zero_grad()

        fake_A = self.fake_As.add_and_sample(self.fake_A, self.opt.batchSize) #opt.batchsize
        self.loss_D_A = self.backward_D(self.disA, self.real_A, fake_A)
        fake_B = self.fake_Bs.add_and_sample(self.fake_B, self.opt.batchSize) #opt.batchsize
        self.loss_D_B = self.backward_D(self.disB, self.real_B, fake_B)

        for group in self.optimizer_D.param_groups:
            for p in group['params']:
                state = self.optimizer_D.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1022):
                        state['step'] = 1000

        self.optimizer_D.step()

        return self.loss_D_A, self.loss_D_B, self.loss_G, self.fake_B, self.cyclic_A, self.fake_A, self.cyclic_B
