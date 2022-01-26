import torch
import torch.utils.data
from torchvision import datasets, transforms




def mnistloader(batchSize):
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
			'./data', train=True, download=True,
			transform=transforms.ToTensor()
		),
		batch_size=batchSize,
		shuffle=True
	)

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
			'./data', train=False, download=True,
			transform=transforms.ToTensor()
		),
		batch_size=batchSize,
		shuffle=True
	)

	return train_loader, test_loader