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


def cifar10loader(batch_size):
    transform = transforms.Compose([transforms.Resize(32), #3*32*32
                                    transforms.ToTensor()
                        ])

    train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(
				root="./data", train=True, download=True, transform=transform),
				batch_size=batch_size,
				shuffle=True)

    test_loader  = torch.utils.data.DataLoader(
					datasets.CIFAR10(
				root="./data", train=False, download=True, transform=transform),
				batch_size=batch_size,
				shuffle=True)
	
    return train_loader, test_loader 
