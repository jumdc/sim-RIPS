import torch
import torchvision.transforms as T
from torchvision import transforms, datasets
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count

class Augment:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """
    def __init__(self, img_size=96, s=1):
        color_jitter = T.ColorJitter(0.8 * s, 
                                    0.8 * s, 
                                    0.8 * s, 
                                    0.2 * s)
        # 10% of the image
        blur = T.GaussianBlur((3, 3), 
                              (0.1, 2.0))
        self.train_transform = T.Compose(
            [T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            # imagenet stats
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.test_transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

def get_stl_dataloader(root, 
                       batch_size, 
                       transform=None,
                       split="unlabeled",
                       num_workers=cpu_count()//2):      
    stl10 = STL10(root, 
                  split=split, 
                  transform=transform, 
                  download=True)
    if split == "unlabeled" or split=="train": 
        # create batches
        train_size = int(0.8 * len(stl10))
        val_size = len(stl10) - train_size      
        train_subset, val_subset = torch.utils.data.random_split(
                                    stl10, 
                                    [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(1))
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_batches = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
        dataloader = {"train": train_dataloader, "val": val_batches}
    else: 
        dataloader = DataLoader(dataset=stl10, 
                               batch_size=batch_size, 
                               num_workers=num_workers)
    return dataloader