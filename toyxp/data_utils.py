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
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        # 10% of the image
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(size=img_size),
                T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
                T.RandomApply([color_jitter], p=0.8),
                T.RandomApply([blur], p=0.5),
                T.RandomGrayscale(p=0.2),
                # imagenet stats
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        self.test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

def get_stl_dataloader(root, 
                       batch_size, 
                       transform=None,
                       split="unlabeled"):
    stl10 = STL10(root, 
                  split=split, 
                  transform=transform, 
                  download=True)
    return DataLoader(dataset=stl10, 
                      batch_size=batch_size, 
                      num_workers=cpu_count()//2)