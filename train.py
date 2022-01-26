import torch 
from torch import optim  
from torch.nn import functional as F
from torchvision.utils import save_image
import math 

from GMVAE_beta import GMVAE

import dataloader as dl 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on GPU")
else:
    device = torch.cuda("cpu:0")
    print("running on CPU")

dev=torch.device("cuda")


train_loader, test_loader = dl.mnistloader(batchSize=64)

gmvae = GMVAE().to(dev)


optimizer = optim.Adam(gmvae.parameters(), lr=1e-3)

K , x_size = 10, 200 

def loss_function(recon_X, X, mu_w, logvar_w, qz,mu_x, logvar_x, mu_px, logvar_px, x_sample):


	N = X.size(0) # batch size

	# 1. Reconstruction Cost = -E[log(P(y|x))]
	# for dataset such as mnist
	
	# for datasets such as tvsum, spiral
	recon_loss = F.binary_cross_entropy(recon_X, X,size_average=False)
	# unpack Y into mu_Y and logvar_Y
	# print(f"shape recon_X : {recon_X.size()}")
	# mu_recon_X, logvar_recon_X = recon_X

    # use gaussian criteria
    # negative LL, so sign is flipped
    # log(sigma) + 0.5*2*pi + 0.5*(x-mu)^2/sigma^2
	# recon_loss = 0.5 * torch.sum(logvar_recon_X + math.log(2*math.pi) + (X - mu_recon_X).pow(2)/logvar_recon_X.exp())

	# 2. KL( q(w) || p(w) )
	KLD_W = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())

	# 3. KL( q(z) || p(z) )
	KLD_Z = torch.sum(qz * torch.log(K * qz + 1e-10))
	

	# 4. E_z_w[KL(q(x)|| p(x|z,w))]
	# KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
	mu_x = mu_x.unsqueeze(-1)
	mu_x = mu_x.expand(-1, x_size, K)

	logvar_x = logvar_x.unsqueeze(-1)
	logvar_x = logvar_x.expand(-1, x_size, K)

	# shape (-1, x_size, K)
	KLD_QX_PX = 0.5 * (((logvar_px - logvar_x) + ((logvar_x.exp() + (mu_x - mu_px).pow(2))/logvar_px.exp())) \
		- 1)

	# transpose to change dim to (-1, x_size, K)
	# KLD_QX_PX = KLD_QX_PX.transpose(1,2)
	qz = qz.unsqueeze(-1)
	qz = qz.expand(-1, K, 1)

	E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz))

	# 5. Entropy criterion
	
	# CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
	# compute likelihood
	
	x_sample = x_sample.unsqueeze(-1)
	x_sample =  x_sample.expand(-1, x_size, K)

	temp = 0.5 * x_size * math.log(2 * math.pi)
	# log likelihood
	llh = -0.5 * torch.sum(((x_sample - mu_px).pow(2))/logvar_px.exp(), dim=1) - 0.5 * torch.sum(logvar_px, dim=1) - temp

	lh = F.softmax(llh, dim=1)

	# entropy
	CV = torch.sum(torch.mul(torch.log(lh+1e-10), lh))
	
	loss = recon_loss + KLD_W + KLD_Z + E_KLD_QX_PX
	
	return loss, recon_loss, KLD_W, KLD_Z, E_KLD_QX_PX, CV


def train(epoch):
	gmvae.train()

	statusString = 'Train epoch: {:5d}[{:5d}/{:5d} loss: {:.6f} ReconL: {:.6f} E(KLD(QX||PX)): {:.6f} CV: {:.6f} KLD_W: {:.6f} KLD_Z: {:.6f}]\n'
	acc = 0.0
	for batch_idx, (data, target) in enumerate(train_loader):
		
		optimizer.zero_grad()

		data = data.to(dev)
		target = target.to(dev)

		mu_x, logvar_x, mu_px, logvar_px, qz, Y , mu_w, logvar_w, \
			x_sample = gmvae(data)

		loss, BCE, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
			= loss_function(Y, data, mu_w, logvar_w, qz,
			mu_x, logvar_x, mu_px, logvar_px, x_sample)

		loss.backward()

		optimizer.step()

		accuracy = 0.0
	
		x_n = qz.argmax(0)
		labels = qz.argmax(1)
		
		for k in range(K):
			counts = torch.sum(((labels==k)+(target==target[x_n[k]]))==2)
			
			
			accuracy = accuracy + counts.item()
		
		acc = acc + accuracy
		accuracy = accuracy / len(data)

		status = statusString.format(epoch, batch_idx+1, len(train_loader),
					loss.item(), BCE.item(), E_KLD_QX_PX.item(),
					CV.item(), KLD_W.item(), KLD_Z.item(), accuracy)
		
		print(status)


# clusters = []
def test(epoch):
	gmvae.eval()

	statusString = 'Test epoch: {:5d}[{:5d}/{:5d} loss: {:.4f} ReconL: {:.4f} E(KLD(QX||PX)): {:.4f} CV: {:.4f} KLD_W: {:.4f} KLD_Z: {:.4f} accuracy: {:.4f}]\n'
	acc = 0.0
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data = data.to(dev)
			
			target = target.to(dev)

			mu_x, logvar_x, mu_px, logvar_px, qz, Y , mu_w, logvar_w, \
				x_sample = gmvae(data)

			loss, BCE, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
				= loss_function(Y, data, mu_w, logvar_w, qz,
				mu_x, logvar_x, mu_px, logvar_px, x_sample)

			accuracy = 0.0
			
			x_n = qz.argmax(0)
			labels = qz.argmax(1)


			for k in range(K):
				counts = torch.sum(((labels==k)+(target==target[x_n[k]]))==2)
				accuracy = accuracy + counts.item()
		
			acc = acc + accuracy
			accuracy = accuracy / len(data)

			status = statusString.format(epoch, batch_idx+1, len(test_loader),
					loss.item(), BCE.item(), E_KLD_QX_PX.item(), CV.item(),
					KLD_W.item(), KLD_Z.item(), accuracy)
			
			print(status)

			
			n = min(data.size(0), 32)
			
			comparision = torch.cat([data[:n], Y[:n]])
			# latent_cluster = torch.cat(qz[:2])
			
			save_image(comparision.cpu(),
					'./results/reconstruction_'+str(epoch)+'.png', nrow=8)
	# 		clusters.append(latent_cluster.cpu())
	# return clusters
						




epochs = 100
for epoch in range(1, epochs+1):
	# train the network
	train(epoch)
	test(epoch)
	if epoch%10:
		torch.save(gmvae.state_dict(), "./models/gmvae"+str(K)+".pth")
	
