from Data.dataset_dict import datasets_dict
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
import random
import numpy as np
import open_clip
from torchvision import datasets
from medmnist import *
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )


class CIFAR10DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["CIFAR10"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["CIFAR10"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class CIFAR10DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["CIFAR10"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["CIFAR10"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class FashionMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["FMNIST"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["FMNIST"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class MNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["MNIST"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["MNIST"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class STL10DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["STL10"](root = "./Data", split = 'train', download = download, transform=self.transform)
        self.test_data = datasets_dict["STL10"](root = "./Data", split = 'test', download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class CelebADataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["CalebA"](root = "./Data", split = 'train', download = download, transform=self.transform)
        self.test_data = datasets_dict["CalebA"](root = "./Data", split = 'test', download = download, transform=self.transform)
        self.valid_data = datasets_dict["CalebA"](root = "./Data", split = 'valid', download = download, transform=self.transform)
        

class SVHNDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["SVHN"](root = "./Data", split = 'train', download = download, transform=self.transform)
        self.test_data = datasets_dict["SVHN"](root = "./Data", split = 'test', download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class QMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["QMNIST"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["QMNIST"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class KMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = datasets_dict["KMNIST"](root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = datasets_dict["KMNIST"](root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        




data_module_dict = {
            "CIFAR10" : CIFAR10DataModule,
            "FMNIST": FashionMNISTDataModule,
            "MNIST": MNISTDataModule,
            "STL10": STL10DataModule,
            #"CalebA": CelebADataModule,
            "SVHN": SVHNDataModule,
            "QMNIST": QMNISTDataModule,
            "KMNIST": KMNISTDataModule,
            }
data_class_dict = {
            "CIFAR10" : 10,
            "FMNIST": 10,
            "MNIST": 10,
            "STL10": 10,
            #"CalebA": CelebADataModule,
            "SVHN": 10,
            "QMNIST": 10,
            "KMNIST": 10,
            }