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
from torch.cuda.amp import autocast
import haste_pytorch as haste
from sklearn.model_selection import train_test_split
from torch.nn.utils import spectral_norm
###########################################################################################################
input_file = 'Producer_start_center_hetrogenous.csv'
batch_size = 50
lambda_gp = 4
lambda_l2 = 1
nb_epoch = int(1e6)
dis_iters = 4
gen_iters = 1
lr_G = 5e-5
lr_D = 4e-5
zoneout = 0.005
dropout = 0.005
test_size = 0.5
###########################################################################################################
num_data = 900

if torch.cuda.is_available(): 
	device = torch.device("cuda")
else: device = torch.device("cpu")

scaler = torch.cuda.amp.GradScaler()

def inf_train_gen(data_gen):
	while True:
		for data, label in data_gen: yield data, label

def load_reals():
	train_size = int(num_data * test_size)
	scaler = MinMaxScaler((-1,1))
	real = pd.read_csv(input_file)
	label = np.array(real[['X_loc','Y_loc']]).reshape(num_data, -1, 2)
	real = np.array(real[['WOPR1','WOPR2','WWPR1','WWPR2']]).reshape(num_data, -1, 4)
	real, real_test, label, label_test = train_test_split(real, label, test_size=0.5, random_state=42)
	real = scaler.fit_transform(real.reshape(-1, real.shape[-1])).reshape(real.shape)
	return scaler, torch.Tensor(real).to(device), torch.LongTensor(label).to(device)

def gen_input(batch_size):
	with torch.no_grad():
		out = torch.empty(batch_size, 2560, 3, device=device)
		x_loc = torch.randint(1, 31, (batch_size, 1), device=device)
		y_loc = torch.randint(1, 31, (batch_size, 1), device=device)

		out[:,:,0] = torch.repeat_interleave(x_loc, repeats=2560, dim=1)
		out[:,:,1] = torch.repeat_interleave(y_loc, repeats=2560, dim=1)
		out[:,:,2] = torch.randn(batch_size, 2560, device=device)
	return out, x_loc, y_loc

def gen_xy(sample_size, x_loc, y_loc):
	with torch.no_grad():
		out = torch.empty(sample_size, 2560, 3, device=device)
		out[:,:,0] = torch.repeat_interleave(torch.tensor(x_loc).view(-1, 1), repeats=2560, dim=1)
		out[:,:,1] = torch.repeat_interleave(torch.tensor(y_loc).view(-1, 1), repeats=2560, dim=1)
		out[:,:,2] = torch.randn(sample_size, 2560, device=device)
	return out

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
		self.n_layers = n_layers
		self.GRU_0 = haste.LayerNormGRU(input_size=in_dim, hidden_size=hidden_dim, zoneout=zoneout, dropout=dropout, batch_first=True)
		self.gru_modules = nn.ModuleList([])
		for _ in range(n_layers - 1): 
			self.gru_modules.append(haste.LayerNormGRU(input_size=hidden_dim, hidden_size=hidden_dim, zoneout=zoneout, dropout=dropout, batch_first=True))
		self.lin = nn.Linear(hidden_dim, out_dim)

	@autocast()
	def forward(self, x):
		h, h_s = self.GRU_0(x)
		for i in range(self.n_layers - 1): h, h_s = self.gru_modules[i](h, state=h_s)
		h = self.lin(h)
		return torch.tanh(h)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
			spectral_norm(nn.Conv1d(4, 16, 6, stride=2)),
			nn.LeakyReLU(inplace=True),
			spectral_norm(nn.Conv1d(16, 64, 6, stride=2)),
			nn.LeakyReLU(inplace=True),
			spectral_norm(nn.Conv1d(64, 256, 6, stride=2)),
			nn.LeakyReLU(inplace=True),
			spectral_norm(nn.Conv1d(256, 128, 6, stride=2)),
			nn.LeakyReLU(inplace=True),
			spectral_norm(nn.Conv1d(128, 64, 6, stride=2)),
			nn.LeakyReLU(inplace=True)
		)
		self.linear_1 = nn.Sequential(
			nn.Linear(4864, 16),
			nn.LayerNorm([16]),
			nn.LeakyReLU(inplace=True),
			nn.Linear(16, 1)
		)
		self.conv_2 = nn.Sequential(
			spectral_norm(nn.Conv1d(64, 32, 6, stride=2)),
			nn.LeakyReLU(inplace=True)
		)
		self.linear_2 = nn.Linear(1152, 30)

		self.conv_3 = nn.Sequential(
			spectral_norm(nn.Conv1d(64, 32, 6, stride=2)),
			nn.LeakyReLU(inplace=True)
		)
		self.linear_3 = nn.Linear(1152, 30)

	@autocast()
	def forward(self, x):
		h = x.view(x.size(0), 4, -1)
		h = self.main(h)

		h_1 = h.reshape(h.size(0), -1)
		h_1 = self.linear_1(h_1)

		h_2 = self.conv_2(h)
		h_2 = h_2.view(h_2.size(0), -1)
		h_2 = self.linear_2(h_2)

		h_3 = self.conv_3(h)
		h_3 = h_3.view(h_3.size(0), -1)
		h_3 = self.linear_3(h_3)
	
		return h_1, h_2, h_3

def gradient_penalty(disc, real_data, fake_data):
	epsilon = torch.rand(batch_size, 1, 1, device=device)
	inter = epsilon * real_data + ((1 - epsilon) * fake_data)
	inter = Variable(inter, requires_grad=True)

	disc_inter, _, _ = disc(inter)
	true_out = torch.ones_like(disc_inter, device=device)

	gp = torch.autograd.grad(outputs=disc_inter, inputs=inter, grad_outputs=true_out, create_graph=True, retain_graph=True)[0]
	gp = gp.reshape(gp.size(0), -1)

	gp = lambda_gp * (((gp + 1e-16).norm(2, dim=1) - 1) ** 2).mean()
	return gp

mm_scaler, real_data, real_label = load_reals()
real_data = data.TensorDataset(real_data, real_label)
real_data = data.DataLoader(real_data, batch_size=batch_size, shuffle=True)
real_data_gen = inf_train_gen(real_data)

gene = Generator(3, 4, 3, 256).to(device)
disc = Discriminator().to(device)

optim_g = optim.Adam(gene.parameters(), lr=lr_G, betas=(0.5,0.999))
optim_d = optim.Adam(disc.parameters(), lr=lr_D, betas=(0.5,0.999))

aux_f = nn.CrossEntropyLoss().to(device)
l2_f = nn.MSELoss().to(device)
hist = np.zeros((1,8))

def train():
	global hist
	gene.train()
	disc.train()	
	
	def disc_train():
		with autocast():
			real_data, real_label = next(real_data_gen)
			real_label_x = real_label[:,0,0]
			real_label_y = real_label[:,1,0]

			z, fake_label_x, fake_label_y = gen_input(batch_size)
			fake_data = gene(z).detach()

			disc_real, disc_real_x, disc_real_y = disc(real_data)
			real_loss = disc_real.mean() + aux_f(disc_real_x, real_label_x.flatten() - 1) + aux_f(disc_real_y, real_label_y.flatten() - 1)
			optim_d.zero_grad()
			scaler.scale(real_loss).backward()
			scaler.step(optim_d)
			scaler.update()

			disc_fake, _, _ = disc(fake_data)
			fake_loss = -disc_fake.mean()
			optim_d.zero_grad()
			scaler.scale(fake_loss).backward()
			scaler.step(optim_d)
			scaler.update()

			if lambda_gp != 0: 
				gp = gradient_penalty(disc, real_data, fake_data)
				optim_d.zero_grad()
				scaler.scale(gp).backward()
				scaler.step(optim_d)
				scaler.update()		
			else: gp = torch.tensor(0, device=device, requires_grad=False)

			d_loss = real_loss.item() + fake_loss.item()
			w_loss = disc_real.mean().item() -disc_fake.mean().item()
			return d_loss, real_loss.item(), -fake_loss.item(), w_loss, gp.item()

	def gen_train():
		with autocast():
			z, fake_label_x, fake_label_y = gen_input(batch_size)
			z = Variable(z, requires_grad=True)
			fake_data = gene(z)

			if lambda_l2 != 0:
				real_data, _ = next(real_data_gen)
				l2_loss = l2_f(real_data, fake_data) * lambda_l2
			else: l2_loss = torch.tensor(0, device=device, requires_grad=False)

			disc_fake, disc_fake_x, disc_fake_y = disc(fake_data)
			g_loss = disc_fake.mean() + aux_f(disc_fake_x, fake_label_x.flatten() - 1) + aux_f(disc_fake_y, fake_label_y.flatten() - 1) + l2_loss 
			optim_g.zero_grad()
			scaler.scale(g_loss).backward()
			scaler.step(optim_g)
			scaler.update()

			return g_loss.item() - l2_loss.item(), l2_loss.item()

	for epoch in range(nb_epoch): 
		start_t = time.time()

		g_losses = AverageMeter()
		d_losses = AverageMeter()
		real_losses = AverageMeter()
		fake_losses = AverageMeter()
		l2_losses = AverageMeter()
		gp_losses = AverageMeter()
		w_losses = AverageMeter()

		for p in disc.parameters(): p.requires_grad_(False)
		for p in gene.parameters(): p.requires_grad_(True)
		for _ in range(gen_iters): 
			g_loss, l2_loss = gen_train()
			g_losses.update(g_loss)
			l2_losses.update(l2_loss)

		for p in disc.parameters(): p.requires_grad_(True)
		for p in gene.parameters(): p.requires_grad_(False)
		for _ in range(dis_iters): 
			d_loss, real_loss, fake_loss, w_loss, gp_loss = disc_train()
			d_losses.update(d_loss)
			real_losses.update(real_loss)
			fake_losses.update(fake_loss)
			w_losses.update(w_loss)
			gp_losses.update(gp_loss)

		torch.cuda.synchronize()
		epoch_time = ((time.time() - start_t))

		print('Epoch: [{0}]\t'
		'Epoch Time {epoch_time:.3f}\t'
		'G Loss {gloss.avg:.4f}\t'
		'D Loss {dloss.avg:.4f}\t'
		'Real Loss {real_losses.avg:.4f}\t'
		'Fake Loss {fake_losses.avg:.4f}\t'
		'W Loss {w_losses.avg:.4f}\t'
		'GP Loss {gp_losses.avg:.4f}\t'
		'L2 Loss {l2_losses.avg:.4f}'.format(
		epoch, epoch_time=epoch_time ,
		gloss=g_losses, dloss=d_losses, real_losses=real_losses, fake_losses=fake_losses, w_losses=w_losses, gp_losses=gp_losses, l2_losses=l2_losses), end=' \n')

		hist = np.concatenate((hist,[[epoch, g_losses.avg, d_losses.avg, real_losses.avg, fake_losses.avg, w_losses.avg, gp_losses.avg, l2_losses.avg]])) 

	np.savetxt('hist.txt', hist[1:], fmt='%.5f')

def eval_gene(gene):
	gene.eval()
	n = 25
	with autocast():
		for x_gen in range(1, 31):
			for y_gen in range(1, 31):
				z = gen_xy(n, x_gen, y_gen)
				fake_data = np.mean(gene(z).detach().cpu().numpy(), 0)
				fake_data = mm_scaler.inverse_transform(fake_data)
				np.savetxt('out_' + str(x_gen) + '_' + str(y_gen) + '.txt', fake_data, fmt='%.4f')

if __name__ == "__main__":
	summary(gene, (2560, 3))
	summary(disc, (2560, 4))
	train()
	eval_gene(gene)
