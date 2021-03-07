import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable 
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torchsummary import summary
from sklearn.preprocessing import StandardScaler, MinMaxScaler
###########################################################################################################
num_data = 800
input_file = 'Producer_start_center_hetrogenous.csv'
batch_size = 50
lambda_gp = 5
lambda_l1 = 3
nb_epoch = int(1e6)
dis_iters = 5
gen_iters = 1
lr_G = 4e-5
lr_D = 2e-5
###########################################################################################################

if torch.cuda.is_available(): 
	device = torch.device("cuda")
else: device = torch.device("cpu")

def inf_train_gen(data_gen):
	while True:
		for data in data_gen: yield data

def load_reals():
	scaler = MinMaxScaler((-1,1))
	real = pd.read_csv(input_file)
	real = np.array(real[['WOPR1','WOPR2','WWPR1','WWPR2']])
	X = np.zeros((num_data,2560,4))
	for i in range(num_data): X[i,:,:] = real[2560*i:2560*(1+i),:]  
	X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
	return scaler, torch.Tensor(X).to(device)

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
	def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256):
		super(Generator, self).__init__()
		self.GRU = nn.GRU(in_dim, hidden_dim, n_layers, batch_first=True)
		self.lin = nn.Linear(hidden_dim, out_dim)

	def forward(self, x):
		h = self.GRU(x)[0]
		h = self.lin(h)
		return torch.tanh(h)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		main = nn.Sequential(
			nn.Conv1d(4, 8, 4, stride=2),
			nn.LayerNorm([8, 1279]),
			nn.LeakyReLU(inplace=True),
			nn.Conv1d(8, 16, 4, stride=2),
			nn.LayerNorm([16, 638]),
			nn.LeakyReLU(inplace=True),
			nn.Conv1d(16, 32, 4, stride=2),
			nn.LayerNorm([32, 318]),
			nn.LeakyReLU(inplace=True)
		)

		self.main = main
		self.linear = nn.Sequential(
			nn.Linear(10176, 8),
			nn.LayerNorm([8]),
			nn.LeakyReLU(inplace=True),
			nn.Linear(8, 1)
		)
	def forward(self, x):
		h = x.view(x.size(0), 4, -1)
		h = self.main(h)
		h = h.view(h.size(0), -1)
		h = self.linear(h)
		return h

def gradient_penalty(disc, real_data, fake_data):
	epsilon = torch.rand(batch_size, 1, 1, device=device)
	inter = epsilon * real_data + ((1 - epsilon) * fake_data)
	inter = Variable(inter, requires_grad=True)

	disc_inter = disc(inter)
	true_out = torch.ones_like(disc_inter, device=device)

	gp = torch.autograd.grad(outputs=disc_inter, inputs=inter, grad_outputs=true_out, create_graph=True, retain_graph=True)[0]
	gp = gp.reshape(gp.size(0), -1)

	gp = lambda_gp * (((gp + 1e-16).norm(2, dim=1) - 1) ** 2).mean()
	return gp

mm_scaler, real_data = load_reals()
real_data = data.TensorDataset(real_data)
real_data = data.DataLoader(real_data, batch_size=batch_size, shuffle=True)
real_data_gen = inf_train_gen(real_data)

gene = Generator(1, 4, 3, 512).to(device)
disc = Discriminator().to(device)

#optim_g = optim.Adam(gene.parameters(), lr=lr_G, betas=(0.5, 0.999))
#optim_d = optim.Adam(disc.parameters(), lr=lr_D, betas=(0.5, 0.999))

optim_g = optim.RMSprop(gene.parameters(), lr=lr_G)
optim_d = optim.RMSprop(disc.parameters(), lr=lr_D)

l1_f = nn.L1Loss().to(device)
hist = np.zeros((1,7))

def train():
	global hist
	gene.train()
	disc.train()	
	
	def disc_train():
		real_data = next(real_data_gen)[0]
		with torch.no_grad(): 
			z = torch.randn(batch_size, 2560, 1, device=device)
			fake_data = gene(z).detach()

		real_loss = disc(real_data).mean()
		optim_d.zero_grad()
		real_loss.backward()
		optim_d.step()

		fake_loss = -disc(fake_data).mean()
		optim_d.zero_grad()
		fake_loss.backward()
		optim_d.step()

		if lambda_gp != 0: 
			gp = gradient_penalty(disc, real_data, fake_data)
			optim_d.zero_grad()
			gp.backward()
			optim_d.step()
		else: gp = torch.tensor(0, device=device, requires_grad=False)

		d_loss = real_loss.item() + fake_loss.item()
		return d_loss, real_loss.item(), -fake_loss.item(), gp.item()

	def gen_train():
		z = Variable(torch.randn(batch_size, 2560, 1, device=device), requires_grad=True)
		fake_data = gene(z)

		if lambda_l1 != 0:
			real_data = next(real_data_gen)[0]
			l1_loss = l1_f(real_data, fake_data) * lambda_l1
		else: l1_loss = torch.tensor(0, device=device, requires_grad=False)

		g_loss = disc(fake_data).mean() + l1_loss
		optim_g.zero_grad()
		g_loss.backward()
		optim_g.step()

		return g_loss.item() - l1_loss.item(), l1_loss.item()

	for epoch in range(nb_epoch): 
		start_t = time.time()

		g_losses = AverageMeter()
		d_losses = AverageMeter()
		real_losses = AverageMeter()
		fake_losses = AverageMeter()
		l1_losses = AverageMeter()
		gp_losses = AverageMeter()

		for p in disc.parameters(): p.requires_grad_(False)
		for p in gene.parameters(): p.requires_grad_(True)
		for _ in range(gen_iters): 
			g_loss, l1_loss = gen_train()
			g_losses.update(g_loss)
			l1_losses.update(l1_loss)

		for p in disc.parameters(): p.requires_grad_(True)
		for p in gene.parameters(): p.requires_grad_(False)
		for _ in range(dis_iters): 
			d_loss, real_loss, fake_loss, gp_loss = disc_train()
			d_losses.update(d_loss)
			real_losses.update(real_loss)
			fake_losses.update(fake_loss)
			gp_losses.update(gp_loss)

		epoch_time = ((time.time() - start_t))

		print('Epoch: [{0}]\t'
		'Epoch Time {epoch_time:.3f}\t'
		'G Loss {gloss.avg:.4f}\t'
		'D Loss {dloss.avg:.4f}\t'
		'Real Loss {real_losses.avg:.4f}\t'
		'Fake Loss {fake_losses.avg:.4f}\t'
		'GP Loss {gp_losses.avg:.4f}\t'
		'L1 Loss {l1_losses.avg:.4f}'.format(
		epoch, epoch_time=epoch_time ,
		gloss=g_losses, dloss=d_losses, real_losses=real_losses, fake_losses=fake_losses, gp_losses=gp_losses, l1_losses=l1_losses), end=' \n')

		hist = np.concatenate((hist,[[epoch, g_losses.avg, d_losses.avg, real_losses.avg, fake_losses.avg, gp_losses.avg, l1_losses.avg]])) 

summary(gene, (2560, 1))
summary(disc, (2560, 4))

train()

with torch.no_grad(): z = torch.randn(1, 2560, 1, device=device)
fake_data = gene(z).detach().cpu().numpy()[0]
fake_data = mm_scaler.inverse_transform(fake_data)
np.savetxt('out_GRU_GP.txt', fake_data, fmt='%.4f')
np.savetxt('hist_GRU_GP.txt', hist[1:], fmt='%.4f')
