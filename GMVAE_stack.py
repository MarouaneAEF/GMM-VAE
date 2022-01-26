import torch
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):

	def __init__(self):
		super(GMVAE, self).__init__()

        self.device = torch.device("cuda")

        self.convStack = nn.Sequential(
        nn.Conv2d(1, 16, 6, 1, 0, bias=False)
		nn.BatchNorm2d(16)
		nn.Conv2d(16, 32, 6, 1, 0, bias=False)
		nn.BatchNorm2d(32)
		nn.Conv2d(32, 64, 5, 2, 1, bias=False)
		nn.BatchNorm2d(64)
        )
        self.fcStack = nn.Sequential(
            nn.Linear(4096, 500, bias=False)
		    nn.BatchNorm1d(500)
        )

        self.mu_x = nn.Linear(500, 200)
		self.logvar_x = nn.Linear(500, 200)

		self.mu_w = nn.Linear(500, 150)
		self.logvar_w = nn.Linear(500, 150)

		self.qz = nn.Linear(500, 10)
		
		# prior generator
		self.h2 = nn.Linear(150, 500) # tanh activation
		# prior x for each cluster
		self.mu_px = nn.ModuleList(
			[nn.Linear(500, 200) for i in range(10)])
		self.logvar_px = nn.ModuleList(
			[nn.Linear(500, 200) for i in range(10)])

        self.decoderFCStack = nn.Sequential(
            nn.Linear(200, 500, bias=False)
		    nn.BatchNorm1d(500)
		    nn.Linear(200, 4096, bias=False)
		    nn.BatchNorm1d(4096)
		    nn.ConvTranspose2d(64, 32, 4, 2, 0, bias=False)
		    nn.BatchNorm2d(32)
        )

        self.decoderConvTStack = nn.Sequential(

        nn.ConvTranspose2d(32, 16, 6, 1, 0, bias=False)
		nn.BatchNorm2d(16)
		nn.ConvTranspose2d(16, 1, 6, 1, 0)

        )

    def encoder(self, x):

        c1 = self.convStack(x)
        c1 = c1.view(-1,4096)
        h1 = self.fcStack(c1)
        
         

