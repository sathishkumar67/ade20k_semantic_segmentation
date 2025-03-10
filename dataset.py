from __future__ import annotations
import torch
from torchvision.transforms import ToTensor 
import numpy as np
from torchvision import datasets
from typing import Callable, Tuple


class CustomCityscapes:
    def __init__(self, root:str, split:str, mode:str='fine', target_type:str='semantic', transforms:Callable=ToTensor(), img_size:Tuple[int, int]=(224, 224)) -> None:
        """
        Initialize the custom Cityscapes dataset container.

        Args:
            root (str): Root directory of the Cityscapes dataset.
            mode (str): Annotation mode ('fine' or 'coarse'). Default is 'fine'.
            target_type (str): Type of target to load ('semantic', 'instance', 'color', etc.). Default is 'semantic'.
        """
        self.root = root
        self.mode = mode
        self.target_type = target_type
        self.split = split
        self.img_size = img_size
        self.transforms = transforms
        self.datasets = {
            'train': datasets.Cityscapes(root, split='train', mode=mode, target_type=target_type),
            'test': datasets.Cityscapes(root, split='test', mode=mode, target_type=target_type),
            'val': datasets.Cityscapes(root, split='val', mode=mode, target_type=target_type),
        }
        self.current_dataset = self.datasets[split]
        self.classes = self.current_dataset.classes
        self.no_of_classes = len(self.classes)

    def __len__(self) -> int:
        """
        Get the nomber of samples in the particular split.

        Returns:
            int: Total number of samples in the particular split.
        """
        return self.current_dataset.__len__()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the particular dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding target with the applied transformations.
        """
        image, target = self.current_dataset.__getitem__(idx)
        image, target = self.transforms(np.array(image.resize(self.img_size))), torch.tensor(np.array(target.resize(self.img_size)), dtype=torch.long)
        return image, target