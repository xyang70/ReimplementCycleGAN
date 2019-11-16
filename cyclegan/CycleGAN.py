import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import random
import itertools
from Generator import Generator
from Discriminator import Discriminator
from torch.autograd import Variable

Tensor = torch.cuda.FloatTensor
#if opt.cuda else torch.Tensor
#Tensor = torch.Tensor

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

        self.criterionGAN = nn.MSELoss()
        self.criterionCycle = nn.L1Loss()

        self.lr_scheduler = None
        self.optimizer_G = optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.opt.lr)
        self.optimizer_D = optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.opt.lr)
        self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.fake_As = ReplayBuffer()
        self.fake_Bs = ReplayBuffer()

    def distributed_update(self, model):
        for param in model.parameters():
            tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
            tensor0 /= float(num_nodes)
            param.grad.data = tensor0.cuda()

    def clip_optimizer_step(self, optimzer):
        for group in self.optimzer.param_groups:
            for p in group['params']:
                state = self.optimzer.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1024):
                        state['step'] = 1000

    def load(self, A, B):
        self.real_A = A
        self.real_B = B

    def forward(self):
        self.fake_B = self.genA2B(self.real_A)
        self.cyclic_A = self.genB2A(self.fake_B)
        self.fake_A = self.genB2A(self.real_B)
        self.cyclic_B = self.genA2B(self.fake_A)
        return self.fake_B, self.cyclic_A, self.fake_A, self.cyclic_B

    def backward_D(self, D, real, fake):
        D_real = D(real)[0]
        loss_D_real = self.criterionGAN(Variable(D_real), Tensor(1).fill_(1.0))

        D_fake = D(fake.detach())[0]
        loss_D_fake = self.criterionGAN(Variable(D_fake), Tensor(1).fill_(0.0))

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        return loss_D

    def optimize_parameters(self):
        self.forward()
        # optimize Generator: calc loss of G -> backward -> update weights
        self.disA.set_grad(False)
        self.disB.set_grad(False)

        self.optimizer_G.zero_grad()
        temp = self.disA(self.fake_B)
        self.loss_genA2B = self.criterionGAN(Variable(self.disA(self.fake_B)[0]), Variable(Tensor(1).fill_(1.0)))
        self.loss_cyclic_A = self.criterionCycle(Variable(self.cyclic_A), Variable(self.real_A))
        self.loss_genB2A = self.criterionGAN(Variable((self.disB(self.fake_A)[0]), Variable((Tensor(1).fill_(1.0)))
        self.loss_cyclic_B = self.criterionCycle(Variable((self.cyclic_B), Variable(self.real_B))
        self.loss_G = self.loss_genA2B + self.loss_genB2A + self.opt.lambd * (self.loss_cyclic_A + self.loss_cyclic_B)  #opt.lambd
        self.loss_G.backward()
        #update the gradient for Generator
        if opt.distributed:
            self.distributed_update(self.genA2B)
            self.distributed_update(self.genB2A)

        self.clip_optimizer_step(self.optimizer_G)

        self.optimizer_G.step()

        # optimize Discriminator: calc loss of D -> backward -> update weights
        self.disA.set_grad(True)
        self.disB.set_grad(True)

        self.optimizer_D.zero_grad()

        fake_A = self.fake_As.add_and_sample(self.fake_A, self.opt.batchSize) #opt.batchsize
        self.loss_D_A = self.backward_D(self.disA, self.real_A, fake_A)
        fake_B = self.fake_Bs.add_and_sample(self.fake_B, self.opt.batchSize) #opt.batchsize
        self.loss_D_B = self.backward_D(self.disB, self.real_B, fake_B)


        if opt.distributed:
            self.distributed_update(self.disA)
            self.distributed_update(self.disB)

        self.clip_optimizer_step(self.optimizer_D)

        self.optimizer_D.step()

        return self.loss_D_A, self.loss_D_B, self.loss_G, self.fake_B, self.cyclic_A, self.fake_A, self.cyclic_B

    def test(self):
        self.fake_B = self.genA2B(self.real_A)
        self.fake_A = self.genB2A(self.real_B)
        return self.disA(self.fake_A)[0],self.disB(self.fake_B)[0]
