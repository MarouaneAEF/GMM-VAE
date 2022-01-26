import torch 
from torch import optim
from trainer import Trainer
import dataloader as dl 
from GMVAE_beta import GMVAE
import warnings
warnings.filterwarnings('ignore')

trainer = Trainer(model=GMVAE(), 
                dataloader=dl.mnistloader(batchSize=64), 
                optim=optim.Adam,
                n_epoches=2,
                )


epochs = 100
for epoch in range(1, epochs+1):
	# train the network
	trainer.train(epoch)
	trainer.test(epoch)
	if epoch%10:
		torch.save(trainer.model.state_dict(), "./models/gmvae"+".pth")