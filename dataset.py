from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
import torch
from torchvision.transforms import ToTensor 
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from datasets import load_dataset

class ADE20KDATASET(Dataset):
    """
    A custom dataset class for the ADE20K dataset, extending PyTorch's Dataset class.
    This class handles loading, transforming, and providing access to the ADE20K dataset
    for training, testing, and validation splits.
    """
    def __init__(self, split: str = "train", transform: Callable = ToTensor(), img_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initialize the ADE20KDATASET class.

        Args:
            split (str): The dataset split to use. Must be one of "train", "test", or "val". Default is "train".
            transform (Callable): A transformation function to apply to the images. Default is ToTensor().
            img_size (Tuple[int, int]): The target size to resize images and masks to. Default is (224, 224).
        """
        self.split = split  # Store the dataset split
        self.transform = transform  # Store the transformation function
        self.img_size = img_size  # Store the target image size
        # Dictionary to hold the dataset splits loaded from the 'scene_parse_150' dataset
        self.datasets = {
            "train": load_dataset("scene_parse_150", split="train", trust_remote_code=True),
            "test": load_dataset("scene_parse_150", split="test", trust_remote_code=True), 
            "val": load_dataset("scene_parse_150", split="validation", trust_remote_code=True),
        }
        # Select the current dataset based on the specified split
        self.current_dataset = self.datasets[split]
        # Calculate the number of samples in the current dataset based on the scene_category length
        self.no_of_samples = len(self.current_dataset["scene_category"])
        # Define the number of classes: 150 classes plus 1 for the background
        self.no_of_classes = 150 + 1

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return self.no_of_samples

    def __getitem__(self, idx):
        """
        Retrieve the image and segmentation mask at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and mask.
        """
        # Load the image, convert to RGB, resize to target size, and apply the transformation
        image = self.transform(transforms.Resize(self.img_size)(self.current_dataset[idx]["image"].convert("RGB")))
        
        # Load the mask, convert to grayscale (L mode), resize using nearest neighbor interpolation,
        # convert to a numpy array, and then to a long tensor
        mask = torch.tensor(np.array(transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST)(self.current_dataset[idx]["annotation"].convert("L"))), dtype=torch.long)
        
        return image, mask
    
    
class CustomCityscapes:
    """
    A custom wrapper class for the Cityscapes dataset, providing a unified interface
    for loading and transforming Cityscapes data for training and validation splits.
    """
    def __init__(self, root: str, split: str, mode: str = 'fine', target_type: str = 'semantic', transforms: Callable = ToTensor(), img_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initialize the CustomCityscapes class.

        Args:
            root (str): Root directory of the Cityscapes dataset.
            split (str): The dataset split to use. Must be one of "train" or "val".
            mode (str): Annotation mode ('fine' or 'coarse'). Default is 'fine'.
            target_type (str): Type of target to load ('semantic', 'instance', 'color', etc.). Default is 'semantic'.
            transforms (Callable): A transformation function to apply to the images. Default is ToTensor().
            img_size (Tuple[int, int]): The target size to resize images and targets to. Default is (224, 224).
        """
        self.root = root  # Store the root directory of the dataset
        self.mode = mode  # Store the annotation mode
        self.target_type = target_type  # Store the type of target to load
        self.split = split  # Store the dataset split
        self.img_size = img_size  # Store the target image size
        self.transforms = transforms  # Store the transformation function
        # Dictionary to hold the dataset splits loaded from torchvision.datasets.Cityscapes
        self.datasets = {
            'train': datasets.Cityscapes(root, split='train', mode=mode, target_type=target_type),
            'val': datasets.Cityscapes(root, split='val', mode=mode, target_type=target_type),
        }
        # Select the current dataset based on the specified split
        self.current_dataset = self.datasets[split]
        # Retrieve the list of classes from the dataset
        self.classes = self.current_dataset.classes
        # Calculate the number of classes based on the length of the classes list
        self.no_of_classes = len(self.classes)

    def __len__(self) -> int:
        """
        Return the total number of samples in the particular split.

        Returns:
            int: Total number of samples in the dataset split.
        """
        return self.current_dataset.__len__()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the image and target at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its corresponding target.
        """
        # Retrieve the image and target from the dataset at the specified index
        image, target = self.current_dataset.__getitem__(idx)
        # Resize the image and target, apply transformations, and convert to tensors
        image = self.transforms(np.array(image.resize(self.img_size)))
        target = torch.tensor(np.array(target.resize(self.img_size)), dtype=torch.long)
        return image, target