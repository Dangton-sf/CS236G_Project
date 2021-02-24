import time
from itertools import repeat
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import autograd
from torch import optim
import torch.nn.functional as F
from torch.utils import data
###########################################################################################################
num_data = 800
input_file = 'Producer_start_center_hetrogenous.csv'
batch_size = 200
lambda_v = 1
nb_epoch = 1000
dis_iters = 1
gen_iters = 100
lr_G = 1e-3
lr_D = 1e-3
###########################################################################################################

if torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")

def inf_train_gen(data_gen):
	while True:
		for data in data_gen: yield data

def load_reals():
	real = pd.read_csv(input_file)
	real = np.array(real[['WOPR1','WOPR2','WWPR1','WWPR2']])
	X = np.zeros((num_data,2560,4))
	for i in range(num_data): X[i,:,:] = real[2560*i:2560*(1+i),:]  
	return torch.Tensor(X).to(device)

class AverageMeter(object):
	def __init__(self): self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		enc = nn.Sequential(
			nn.Conv2d(1, 64, (3,3)),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d((2, 2)),
			nn.Conv2d(64, 32, (3,3)),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.1)
		)

		dec = nn.Sequential(
			nn.ConvTranspose1d(32, 32, 3, stride=3),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.1),
			nn.ConvTranspose1d(32, 16, 3, stride=3),
			nn.BatchNorm1d(16),
			nn.LeakyReLU(0.1),
			nn.ConvTranspose1d(16, 4, 3, stride=2)
		)

		self.enc = enc
		self.dec = dec

	def forward(self, x):
		h = self.enc(x)
		h = h.view(h.size(0), 32, -1)
		h = self.dec(h)
		h = h[:, :, :2560]
		h = h.reshape(h.size(0), -1, 4)
		return h

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		main = nn.Sequential(
			nn.Conv1d(4, 8, 3, stride=4),
			nn.LayerNorm([8, 640]),
			nn.LeakyReLU(0.1),
			nn.Conv1d(8, 16, 3, stride=4),		
			nn.LayerNorm([16, 160]),
			nn.LeakyReLU(0.1)
		)

		self.main = main
		self.linear = nn.Linear(2560, 1)

	def forward(self, x):
		h = x.view(x.size(0), 4, -1)
		h = self.main(h)
		h = h.view(h.size(0), -1)
		h = self.linear(h)
		return h

def gradient_penalty(disc, real_data, fake_data):
	epsilon = torch.rand(batch_size, 1, 1).to(device)
	inter = epsilon * real_data + ((1 - epsilon) * fake_data)
	inter = autograd.Variable(inter, requires_grad=True)

	disc_inter = disc(inter)
	true_out = torch.ones_like(disc_inter).to(device)

	gp = autograd.grad(outputs=disc_inter, inputs=inter, grad_outputs=true_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
	gp = gp.view(gp.size(0), -1)

	gp = lambda_v * torch.mean((gp.norm(2, dim=1) - 1) ** 2)
	return gp

real_data = data.TensorDataset(load_reals())
real_data = data.DataLoader(real_data, batch_size=batch_size, shuffle=True)
real_data_gen = inf_train_gen(real_data)

gene = Generator().to(device)
disc = Discriminator().to(device)

optim_g = optim.Adam(gene.parameters(), lr=lr_G)
optim_d = optim.Adam(disc.parameters(), lr=lr_D)

def train():

	gene.train()
	disc.train()	

	def disc_train():
		optim_d.zero_grad()
		real_data = next(real_data_gen)[0]
		z = autograd.Variable(torch.randn(batch_size, 1, 30, 30).to(device), requires_grad=True)

		fake_data = gene(z)
		gp = gradient_penalty(disc, real_data, fake_data)
		real_loss = disc(real_data).mean()
		fake_loss = disc(fake_data).mean()
		d_loss = real_loss - fake_loss + gp
		d_loss.backward()
		optim_d.step()

		return d_loss.item(), real_loss.item(), fake_loss.item(), gp.item()

	def gen_train():
		optim_g.zero_grad()

		z = autograd.Variable(torch.randn(batch_size, 1, 30, 30).to(device), requires_grad=True)
		fake_data = gene(z)

		g_loss = disc(fake_data).mean()
		g_loss.backward()
		optim_g.step()

		return g_loss.item()	

	for epoch in range(nb_epoch): 
		start_t = time.time()

		g_losses = AverageMeter()
		d_losses = AverageMeter()
		real_losses = AverageMeter()
		fake_losses = AverageMeter()
		gp_losses = AverageMeter()

		for _ in range(gen_iters): 
			g_loss = gen_train()
			g_losses.update(g_loss)

		for _ in range(dis_iters): 
			d_loss, real_loss, fake_loss, gp = disc_train()
			d_losses.update(d_loss)
			real_losses.update(real_loss)
			fake_losses.update(fake_loss)
			gp_losses.update(gp)

		torch.cuda.synchronize()
		epoch_time = ((time.time() - start_t))

		print('Epoch: [{0}]\t'
		'Epoch Time {epoch_time:.3f}\t'
		'G Loss {gloss.avg:.4f}\t'
		'D Loss {dloss.avg:.4f}\t'
		'Real Loss {real_losses.avg:.4f}\t'
		'Fake Loss {fake_losses.avg:.4f}'.format(
		epoch, epoch_time=epoch_time ,
		gloss=g_losses, dloss=d_losses, real_losses=real_losses, fake_losses=fake_losses), end=' \n')

train()
