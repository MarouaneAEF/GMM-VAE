import torch
from torch import nn, relu
from torch.nn import functional as F

class GMVAE(nn.Module):

	def __init__(self, K=10, x_size=200, hidden_size=500, w_size=150):
		super(GMVAE, self).__init__()
		self.K = K
		self.x_size = x_size
		self.hiddend_size = hidden_size
		self.w_size = w_size
		# cuda
		self.device = torch.device("cuda")
		# Recognition model:
        # input stack :
		# ----------encConvStack----------
		self.encConvStack = nn.Sequential(
			nn.Conv2d(1, 16, 6, 1, 0, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),

			nn.Conv2d(16, 32, 6, 1, 0, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 64, 5, 2, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)
		
		# --- output of bottlneck -----
		x = torch.rand(1,28,28).view(-1,1,28,28)
		self._after_bottlneck = None
		self._before_bottlneck = None
		self.before_bottlneck(x)
        # ----- we've got a flattned tensor---
		
		# --------------encFcStack--------------
		self.fcStack = nn.Sequential(
			nn.Linear(self._after_bottlneck, self.hiddend_size, bias=False),
			nn.BatchNorm1d(self.hiddend_size),
			nn.ReLU()
		)
		
		# -------------------- encoder outputs --------------------
		self.mu_x = nn.Linear(self.hiddend_size, self.x_size)
		self.logvar_x = nn.Linear(self.hiddend_size, self.x_size)

		self.mu_w = nn.Linear(self.hiddend_size, self.w_size)
		self.logvar_w = nn.Linear(self.hiddend_size, self.w_size)

		self.qz = nn.Linear(self.hiddend_size, self.K)
		#  -------------------------------------------------------

		# prior generator
		# ----------priorStack-------------
		self.priorStack = nn.Sequential(
			nn.Linear(self.w_size, self.hiddend_size),
			nn.Tanh()
		)
		
		# -----------------------prior x for each cluster---------------------
		self.mu_px = nn.ModuleList(
			[nn.Linear(self.hiddend_size, self.x_size) for i in range(self.K)])
		self.logvar_px = nn.ModuleList(
			[nn.Linear(self.hiddend_size, self.x_size) for i in range(self.K)])
		# ------------------------------------------

		# generative model
		# -----------------------decFcStack------------------------
		self.decFcStack = nn.Sequential(

			nn.Linear(self.x_size, self._after_bottlneck, bias=False),
			nn.BatchNorm1d(self._after_bottlneck),
			nn.ReLU(),
		)
		# ----------------------deconvStack------------------------ 
		self.deconvStack = nn.Sequential(
			nn.ConvTranspose2d(64, 32, 4, 2, 0, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.ConvTranspose2d(32, 16, 6, 1, 0, bias=False),
			nn.BatchNorm2d(16),
			nn.ReLU(),

			nn.ConvTranspose2d(16, 1, 6, 1, 0)
		)
		
	

	def before_bottlneck(self, x):

		x = self.encConvStack(x) 
		if self._before_bottlneck is None:
			self._before_bottlneck = x[0].shape
		if self._after_bottlneck is None:
			self._after_bottlneck = x[0].flatten().shape[0] 

		return x 



	def encode(self, x):
		
		x = self.before_bottlneck(x)
		x = x.view(-1, self._after_bottlneck)
		# print(f"---------------------self._after_bottlneck: {self._after_bottlneck}")
		
		x = self.fcStack(x)
        # q(z|y)
		qz = F.softmax(self.qz(x), dim=1)
        # q(x|y)
		mu_x = self.mu_x(x)
		logvar_x = self.logvar_x(x)
        # q(w|y)
		mu_w = self.mu_w(x)
		logvar_w = self.logvar_w(x)

		return qz, mu_x, logvar_x, mu_w, logvar_w

	def priorGenerator(self, w_sample):
        #  P(x|z, w) generation for all z_i 
		batchSize = w_sample.size(0)

        #  network beta 
		h = self.priorStack(w_sample)
        # p(x|z,w)
		mu_px = torch.empty(batchSize, self.x_size, self.K,
			 device=self.device, requires_grad=False)
		logvar_px = torch.empty(batchSize, self.x_size, self.K,
			 device=self.device, requires_grad=False)

		for i in range(self.K):
			mu_px[:, :, i] = self.mu_px[i](h)
			logvar_px[:, :, i] = self.logvar_px[i](h)

		return mu_px, logvar_px

	def decoder(self, x_sample):
		
		h = self.decFcStack(x_sample)
		# print(f"-------------self._before_bottlneck: {self._before_bottlneck}")
		h = h.view(-1, *self._before_bottlneck)
		h = self.deconvStack(h)
		# Y = F.sigmoid(h)
		Y = torch.sigmoid(h)
		return Y

	def reparameterize(self, mu, logvar):
		'''
		compute z = mu + std * epsilon
		'''
		if self.training:
			# do this only while training
			# compute the standard deviation from logvar
			std = torch.exp(0.5 * logvar)
			# sample epsilon from a normal distribution with mean 0 and
			# variance 1
			eps = torch.randn_like(std)
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, X):
		qz, mu_x, logvar_x, mu_w, logvar_w = self.encode(X)

		w_sample = self.reparameterize(mu_w, logvar_w)
		x_sample = self.reparameterize(mu_x, logvar_x)

		mu_px, logvar_px = self.priorGenerator(w_sample)
		Y = self.decoder(x_sample)

		return mu_x, logvar_x, mu_px, logvar_px, qz, Y, mu_w, \
			logvar_w, x_sample