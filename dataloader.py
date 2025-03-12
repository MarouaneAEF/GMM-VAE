import torch
import torch.utils.data
from torchvision import datasets, transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob




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


class CustomHighResImageDataset(Dataset):
    """
    Custom dataset for loading high resolution images from a directory without downsampling
    """
    def __init__(self, img_dir, transform=None, split='train', train_ratio=0.8, max_images=None, random_sample=False):
        self.img_dir = img_dir
        self.transform = transform
        
        # Get all image files with common image extensions
        pattern = os.path.join(img_dir, "**/*.*")
        self.image_files = []
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            self.image_files.extend(glob.glob(os.path.join(img_dir, f"*{ext}"), recursive=True))
            self.image_files.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}"), recursive=True))
        
        # Make paths relative to img_dir for easier debugging
        self.image_files = sorted(self.image_files)
        
        if len(self.image_files) == 0:
            print(f"WARNING: No images found in {img_dir}")
            print(f"Searched for extensions: .jpg, .jpeg, .png, .bmp, .tiff")
            # Try listing all files in the directory to debug
            all_files = glob.glob(os.path.join(img_dir, "*"))
            print(f"All files in directory: {all_files}")
        else:
            print(f"Found {len(self.image_files)} images in {img_dir}")
        
        # Limit number of images if max_images is specified
        if max_images is not None and max_images > 0 and max_images < len(self.image_files):
            if random_sample:
                # Randomly sample max_images from the dataset
                import random
                print(f"Randomly sampling {max_images} images from dataset of size {len(self.image_files)}")
                random.seed(42)  # For reproducibility
                self.image_files = random.sample(self.image_files, max_images)
            else:
                print(f"Limiting dataset to first {max_images} images (from total of {len(self.image_files)})")
                self.image_files = self.image_files[:max_images]
        
        # Split into train and test
        split_idx = int(len(self.image_files) * train_ratio)
        
        if split == 'train':
            self.image_files = self.image_files[:split_idx]
        else:  # test
            self.image_files = self.image_files[split_idx:]
            
        # Get image size from first image if available
        if len(self.image_files) > 0:
            first_img_path = self.image_files[0]
            try:
                with Image.open(first_img_path) as img:
                    self.img_width, self.img_height = img.size
                    print(f"Sample image dimensions: {self.img_width}×{self.img_height}")
            except Exception as e:
                print(f"Error reading first image: {e}")
                self.img_width, self.img_height = 256, 256  # Fallback default
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Open image with PIL
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform to tensor only (no resize)
                image = transforms.ToTensor()(image)
            
            # Return dummy label 0 (since we're doing unsupervised learning)
            return image, 0
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a small blank image as fallback
            return torch.zeros(3, 32, 32), 0


def custom_dataloader(img_dir, batch_size, preserve_size=True, max_images=None, random_sample=False, target_size=None):
    """
    Create dataloader for custom high-resolution image dataset
    
    Args:
        img_dir: Path to directory containing images
        batch_size: Batch size for training
        preserve_size: If True, keep original image dimensions (ignored if target_size is provided)
        max_images: Maximum number of images to use (None for all images)
        random_sample: If True, randomly sample max_images instead of taking the first ones
        target_size: Optional tuple (width, height) to resize images to a specific resolution
    """
    # Create transforms based on whether we're preserving size or resizing
    if target_size is not None:
        print(f"Resizing all images to {target_size[0]}×{target_size[1]}")
        transform = transforms.Compose([
            transforms.Resize((target_size[1], target_size[0])),  # (height, width)
            transforms.ToTensor(),
        ])
    else:
        # Just convert to tensor, preserving original dimensions
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # Create datasets
    train_dataset = CustomHighResImageDataset(img_dir, transform=transform, split='train', 
                                             max_images=max_images, random_sample=random_sample)
    test_dataset = CustomHighResImageDataset(img_dir, transform=transform, split='test', 
                                            max_images=max_images, random_sample=random_sample)
    
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in the training set from {img_dir}")
    
    # Create dataloaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    
    # Get image dimensions - use target_size if provided, otherwise from dataset
    if target_size is not None:
        img_width, img_height = target_size
    else:
        img_height = getattr(train_dataset, 'img_height', 256)
        img_width = getattr(train_dataset, 'img_width', 256)
    
    return train_loader, test_loader, (img_height, img_width)
