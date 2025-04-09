import os

from dataclasses import dataclass

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from sampler import MintermSampler

from typing import Tuple

import logging

@dataclass
class Batch:
    images: Tensor = None
    labels: Tensor = None

class CIFARTransform(object):
    def __init__(self, split: str) -> None:
        mu  = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        if split.lower() == "train":
            self.transform = T.Compose([T.RandomCrop(32, padding=4),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor(),
                                        T.Normalize(mu,std)])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mu,std)])
            
    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x
    
class MNISTTransform(object):
    def __init__(self, split: str) -> None:
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize((0.1307,), (0.3081,))])
            
    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x
    
class FashionMNISTTransform(object):
    def __init__(self, split: str) -> None:
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize((0.5,), (0.5,))])
            
    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x

class CELEBATransform(object):
    def __init__(self, split: str, img_size: int) -> None:
        if split.lower() == "train":
            self.transform = T.Compose([T.Resize(img_size),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor()])
        else:
            self.transform = T.Compose([T.Resize(img_size),
                                        T.ToTensor()])

    def __call__(self, sample: Image) -> Tensor:
        x = self.transform(sample)
        return x


class CELEBA(Dataset):
    def __init__(
        self,
        root : str,
        split : str,
        class_select=None,
        transform=None
    ) -> None:
        super(CELEBA, self).__init__()

        self.root = root
        self.split = split.lower()
        self.transform = transform

        self.splits = {
            "train" : 0,
            "val"   : 1,
            "test"  : 2,
        }

        split_im_filenames = set()
        with open(os.path.join(self.root, "list_eval_partition.txt")) as f:
            for line in f:
                im_filename, split = line.split()
                if int(split) == self.splits[self.split]:
                    split_im_filenames.add(im_filename)

        self.im_filenames = []
        self.targets = []

        with open(os.path.join(self.root, "list_attr_celeba.txt")) as f:
            for i, line in enumerate(f):
                data = line.rstrip().split()
                if i == 1:
                    self.classes = data
                    self.class_to_idx = {c : i for i, c in enumerate(self.classes)}
                    if class_select is not None:
                        self.classes = class_select
                if i > 1:
                    im_filename = data[0]
                    if im_filename in split_im_filenames:
                        targets = [max(int(data[1+self.class_to_idx[c]]), 0) for c in self.classes]
                        if sum(targets) > 0:
                            self.im_filenames.append(im_filename)
                            self.targets.append(targets)

        self.class_to_idx = {t : i for i, t in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.im_filenames)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        im_filename = self.im_filenames[idx]
        im_path = os.path.join(self.root, "img_align_celeba", im_filename)
        im = Image.open(im_path)
        im = im.crop((0, 40, im.size[0], im.size[1]))

        if self.transform:
            im = self.transform(im)

        return im, torch.tensor(self.targets[idx], dtype=torch.int32)

def collate_fn(batch) -> Batch:
    labels = torch.stack([b[1] for b in batch], dim=0)
    images = torch.stack([b[0] for b in batch], dim=0)
    return Batch(images=images, labels=labels)

def get_loader(
    dataset_name: str,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    split = split.lower()
    dataset_name = dataset_name.upper()

    cifar10_root = "/mnt/localdisk/gabriel/nodocker/CIFAR10"
    cifar100_root = "/mnt/localdisk/gabriel/nodocker/CIFAR100"
    celeb_a_root = "/mnt/localdisk/gabriel/nodocker/CELEBA"
    mnist_root = "/mnt/localdisk/gabriel/nodocker/MNIST"
    fashionmnist_root = "/mnt/localdisk/gabriel/nodocker/FashionMNIST"

    if dataset_name == "MNIST":
        dataset = MNIST(
            root=mnist_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda i : F.one_hot(torch.tensor(i), num_classes=10),
            transform=MNISTTransform(split))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

    elif dataset_name == "FASHIONMNIST":
        dataset = FashionMNIST(
            root=fashionmnist_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda i : F.one_hot(torch.tensor(i), num_classes=10),
            transform=FashionMNISTTransform(split)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

    elif dataset_name == "CIFAR10":
        dataset = CIFAR10(
            root=cifar10_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda i : F.one_hot(torch.tensor(i), num_classes=10),
            transform=CIFARTransform(split),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

    elif dataset_name == "CIFAR100":
        dataset = CIFAR100(
            root=cifar100_root,
            train=(split == "train"),
            download=True,
            target_transform=lambda i : F.one_hot(torch.tensor(i), num_classes=100),
            transform=CIFARTransform(split),
        )

        if split == "train":
            dataloader = DataLoader(
                dataset,
                batch_sampler=MintermSampler(dataset.targets, batch_size, 100, 50),
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )

    elif dataset_name == "CELEBA":
        dataset = CELEBA(
            root=celeb_a_root,
            class_select=["Bald",
                          "Eyeglasses",
                          "Wearing_Necktie",
                          "Wearing_Hat",
                          "Male"],
            split=split, 
            transform=CELEBATransform(split=split, img_size=img_size),
        )

        if split == "train":
            dataloader = DataLoader(
                dataset,
                batch_sampler=MintermSampler(dataset.targets, batch_size, 5, 5),
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
    else:
        raise ValueError(f"Unkown dataset {dataset_name}")
            
    logging.info(f"Loaded {dataset_name}/{split} with {len(dataset)} samples")

    return dataloader
