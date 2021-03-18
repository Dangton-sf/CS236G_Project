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
	gene.load_state_dict(torch.load('gene_center_hetrogenous.zip'))
	eval_gene(gene)
