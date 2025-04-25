import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torch.utils.data import Subset
from torchvision.transforms import v2, ToTensor
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
import glob
from multiDist.utils.embedder_info import get_embedder
import torchvision.models as models
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics.functional import auroc
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import BinaryAccuracy
from transformers import DPTModel, PvtV2ForImageClassification, ViTHybridModel, CvtModel, LevitConfig, LevitModel, AutoImageProcessor, AutoModel, AutoFeatureExtractor, SwinForImageClassification, MobileViTFeatureExtractor, MobileViTForImageClassification, ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, DeiTForImageClassificationWithTeacher, BeitImageProcessor, BeitForImageClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import random

def get_checkpoint(checkpoint, index = 0):
    prefix = f"embedders.{index}."
    new_ckpt = {
        k[len(prefix):]: v
        for k, v in checkpoint.items()
        if k.startswith(prefix)
    }
    return new_ckpt

def get_trasform_vision(args):
    #!UPDATE!#
    train_transforms_list = [v2.Resize((256, 256)), v2.CenterCrop((224, 224))]
    '''augmentations = args.augmentations
    if augmentations.get("ColorJitter"):
        train_transforms_list.append(
            v2.ColorJitter(
                brightness=augmentations["ColorJitter"].get("brightness", 0.2),
                contrast=augmentations["ColorJitter"].get("contrast", 0.2),
                saturation=augmentations["ColorJitter"].get("saturation", 0.2),
                hue=augmentations["ColorJitter"].get("hue", 0.2),
            )
        )
    if augmentations.get("RandomHorizontalFlip", False):
        train_transforms_list.append(v2.RandomHorizontalFlip())
    if augmentations.get("RandomRotation"):
        train_transforms_list.append(
            v2.RandomRotation(augmentations["RandomRotation"])
        )
    if augmentations.get("GaussianBlur"):
        train_transforms_list.append(
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=augmentations["GaussianBlur"].get("kernel_size", 3))],
                p=augmentations["GaussianBlur"].get("probability", 0.2),
            )
        )'''
    train_transforms_list.append(v2.ToTensor())
    return v2.Compose(train_transforms_list)



def get_cub_train(root, train = True, download = False, transform = None):
    shuffle = False
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #transforms.RandomErasing(p=0.25, value='random')
        ])
        all_data = datasets.ImageFolder(root+'/CUB_200_2011/images', transform=transform)
        train_indices = []
        test_indices = []
        
        with open('./Data/train_test_split.txt', 'r') as f:
            for line in f:
                index, is_train = map(int, line.strip().split())
                if is_train == 0:
                    test_indices.append(index)
                elif is_train == 1:
                    train_indices.append(index)
        
        # Create subsets for train and test
        train_data = Subset(all_data, train_indices)
        test_data = Subset(all_data, test_indices)
        
        return train_data
    
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        all_data = datasets.ImageFolder(root+'/CUB_200_2011/images', transform=transform)
        train_indices = []
        test_indices = []
        
        with open('./Data/train_test_split.txt', 'r') as f:
            for line in f:
                index, is_train = map(int, line.strip().split())
                if is_train == 0:
                    test_indices.append(index)
                elif is_train == 1:
                    train_indices.append(index)
        
        # Create subsets for train and test
        train_data = Subset(all_data, train_indices)
        test_data = Subset(all_data, test_indices)
        return test_data
def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes
def get_data_loaders(data_dir, batch_size, train = False, shuffle = False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #transforms.RandomErasing(p=0.25, value='random')
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data)*0.75)
        valid_data_len = int((len(all_data) - train_data_len)/2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        return train_loader, train_data_len
    
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data)*0.70)
        valid_data_len = int((len(all_data) - train_data_len)/2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        return (val_loader, test_loader, valid_data_len, test_data_len)


datasets_dict = {
            "cub": get_cub_train,
            #"Food101": torchvision.datasets.Food101,
            "DTD": torchvision.datasets.DTD,
            #"ImageNet": torchvision.datasets.ImageNet,
            "FGVCAircraft": torchvision.datasets.FGVCAircraft,
            "CIFAR10" : torchvision.datasets.CIFAR10,
            #"CIFAR100": torchvision.datasets.CIFAR100,
            #"FakeData": torchvision.datasets.FakeData,#a random data
            #"FMNIST": torchvision.datasets.FashionMNIST,
            #"Flickr8": torchvision.datasets.Flickr8k,#needs to be downloaded manually
            #"Flickr30": torchvision.datasets.Flickr30k,#needs to be downloaded manually
            #"ImageNet": torchvision.datasets.ImageNet,#RuntimeError: The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupted. You need to download it externally and place it in ./Data.
            #"LSUN": torchvision.datasets.LSUN,#needs to be downloaded manually
            #"MNIST": torchvision.datasets.MNIST,
            #"Places365": torchvision.datasets.Places365,too big
            "STL10": torchvision.datasets.STL10,#(image, target) where target is index of the target class.
            #"CelebA": torchvision.datasets.CelebA,
            "SVHN": torchvision.datasets.SVHN,#	(image, target) where target is index of the target class.
            #"QMNIST": torchvision.datasets.QMNIST,#?
            #"EMNIST": torchvision.datasets.EMNIST,#?TypeError: EMNIST.__init__() missing 1 required positional argument: 'split'
            #"KMNIST": torchvision.datasets.KMNIST,#?
            #"Omniglot": torchvision.datasets.Omniglot,#?
            #"CityScapes": torchvision.datasets.Cityscapes,
            #"COCO": torchvision.datasets.CocoCaptions,#Tuple (image, target). target is a list of captions for the image.
            #"SBU": torchvision.datasets.SBU, #image, caption
            #"Detection": torchvision.datasets.CocoDetection,
            #"PhotoTour": torchvision.datasets.PhotoTour,
            #"SBD": torchvision.datasets.SBDataset,
            #"USPS": torchvision.datasets.USPS,
            #"VOC": torchvision.datasets.VOCDetection, #(image, target) where target is a dictionary of the XML tree.
            }


class TeacherEmbeddingDataset(Dataset):
    def __init__(self, teacher_paths, teacher_name, datasets, length, indices):
        self.teacher_paths = teacher_paths
        self.teacher_name = teacher_name
        self.length = length
        index = 0
        self.embeding_index_to_file = {}
        for ds in datasets:
            data = np.load(self.teacher_paths + "/"+teacher_name+"_"+ds+".npy", mmap_mode="r")
            self.embeding_index_to_file[index] = self.teacher_paths + "/"+teacher_name+"_"+ds+".npy"
            index = index + np.shape(data)[0]
        self.indices = indices

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        index = np.argmax(self.indices > idx) - 1
        ds_index = self.indices[index]
        local_index = idx - ds_index
        data = np.load(self.embeding_index_to_file[ds_index], mmap_mode="r")
        return data[local_index]
        

        
class VisionDataset:

    def __init__(  self,  teachers_path = "./Embeddings/vision", list_teachers = None, transform = None):
        self.teacher_paths = teachers_path
        index = 0
        teacher_name = list_teachers[0]
        self.embeding_index_to_ds = {}
        self.datasets = []
        for ds in datasets_dict.keys():
            self.datasets.append(ds)
            self.embeding_index_to_ds[index] = datasets_dict[ds](root = "./Data",download = True, transform=transform)
            data = np.load(self.teacher_paths + "/"+teacher_name+"_"+ds+".npy", mmap_mode="r")
            index = index + np.shape(data)[0]
        self.length = index
        self.teachers = []
        self.teachers_ds = []
        self.indices = np.array(list(self.embeding_index_to_ds.keys()))
        self.list_teachers = list_teachers
        for teacher_name in list_teachers:
            self.teachers.append(get_embedder(teacher_name))
            self.teachers_ds.append(TeacherEmbeddingDataset(self.teacher_paths, teacher_name, self.datasets, self.length, self.indices))
        
    
    def __len__(self):
        return self.length
    
    '''Returns image, teacher embeddings, teacher indices'''
    def __getitem__(self, idx):
        index = np.argmax(self.indices > idx) - 1
        ds_index = self.indices[index]
        local_index = idx - ds_index
        img, lable = self.embeding_index_to_ds[ds_index][local_index]
        emb = []#torch.tensor([])
        for teacher in self.teachers_ds:
            emb.append(teacher[idx])# = torch.cat((emb, teacher[idx]))
        return np.array(img), emb, list(range(len(self.list_teachers)))

        



def build_fc_layers(dim, output_dim, num_classes, norm=""):
    """
    Builds an nn.Sequential FC layer stack based on the configuration.

    Args:
        fc_layers_config (list): List of layer configurations. Each configuration is a dict with:
            - in_features: Input dimension of the layer.
            - out_features: Output dimension of the layer.
            - activation: (Optional) Activation function (e.g., "ReLU").
            - dropout: (Optional) Dropout probability.
        output_dim (int): Dimension of the input to the first FC layer.
        num_classes (int): Dimension of the output from the final FC layer.

    Returns:
        nn.Sequential: The constructed FC layer stack.
    """
    if norm == "":
        norm_fn = nn.Identity
    elif norm == "batch":
        norm_fn = nn.BatchNorm1d
    elif norm=="layer":
        norm_fn = nn.LayerNorm
    else:
        raise NotImplementedError()


    layers = []
    # Handle the default case where no configuration is provided
    if dim == 0:
        print("fully connected ok")
        layers.append(norm_fn(output_dim))
        layers.append(nn.Linear(output_dim, num_classes))
        return nn.Sequential(*layers)
    else:
        return nn.Sequential(
                nn.Linear(output_dim, dim),  # Hidden size from Swin backbone
                nn.ReLU(),
                norm_fn(dim),
                #nn.Dropout(0.5),
                nn.Linear(dim, num_classes)
            )
    # Dynamically build the layers
    for i, layer in enumerate(fc_layers_config):
        # Set in_features for the first layer
        in_features = layer.get("in_features", output_dim if i == 0 else None)
        # Set out_features for the last layer
        out_features = layer.get("out_features", num_classes if i == len(fc_layers_config) - 1 else None)

        if in_features is None or out_features is None:
            raise ValueError("Each layer configuration must specify 'in_features' and 'out_features'.")

        layers.append(nn.Linear(in_features, out_features))

        # Add optional activation function
        if layer.get("activation"):
            layers.append(getattr(nn, layer["activation"])())

        # Add optional dropout layer
        if layer.get("dropout", 0) > 0:
            layers.append(nn.Dropout(layer["dropout"]))

    # Ensure the output of the last layer matches num_classes
    if fc_layers_config[-1]["out_features"] != num_classes:
        layers.append(nn.Linear(fc_layers_config[-1]["out_features"], num_classes))

    return nn.Sequential(*layers)

class TunnerModel(L.LightningModule):
    def __init__(
        self,
        model = models.resnet18(pretrained=True),
        output_dim = 512,
        lr=0.001, momentum=0.9, nesterov = True, weight_decay = 1, batch_size = 64, num_classes = 100, fc_layers_config = None, scheduler = False, norm=""
    ):
        
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.scheduler = scheduler        
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = build_fc_layers(fc_layers_config, output_dim, num_classes, norm=norm)
        '''nn.Sequential(
                #nn.Linear(output_dim, num_classes),  # Hidden size from Swin backbone
                #nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(output_dim, num_classes)
            )'''
        print(num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        if num_classes == 2:
            self.metric = BinaryAccuracy()
        else:
            self.metric = MulticlassAccuracy(num_classes=num_classes)

        self.predictions = []
        self.targets = []

        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []
        self.train_loss = []
        self.val_loss = []
        
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        if isinstance(x, tuple):
            x = x[-1]
        out = self.fc(x)
        return out
    
    def compute_loss(self, predictions, targets):
        """
        Handles both soft and hard labels for computing loss.
        """
        if targets.ndim == 2:  # Soft labels
            loss = F.kl_div(
                F.log_softmax(predictions, dim=1),  # Log probabilities of predictions
                targets,  # Soft labels
                reduction="batchmean",
            )
        else:  # Hard labels
            loss = F.cross_entropy(predictions, targets)
        return loss
    
    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum, nesterov = self.nesterov, weight_decay = self.weight_decay)
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)#
        scheduler = CosineAnnealingLR(optimizer, eta_min = 5e-5, T_max = 100)
        if self.scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}#
        return {"optimizer": optimizer}
    
    def process_batch(self, batch):
        img = batch[0]#.to(self.device)
        lab = batch[1]#.to(self.device)
        if img.shape[1] < 3:#?
            img = torch.stack((img,img,img), dim = 1).squeeze(2)
        out = self.forward(img)
        prd = torch.softmax(out, dim=1)
        loss = self.compute_loss(prd, lab)
        return loss, prd, lab

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.train_step_preds.append(prd)
        self.train_step_trgts.append(lab)
        self.log('train_loss', loss, 
            on_step=True,
            on_epoch=False,
            prog_bar=True, batch_size=self.batch_size)        
        '''batch_ratio = len(np.where(lab.cpu().numpy() == 1)[0]) / len(np.where(lab.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=self.batch_size)                        
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)'''
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)
        all_trgts = torch.cat(self.train_step_trgts, dim=0)

        # Compute metrics using hard labels
        hard_targets = all_trgts.argmax(1) if all_trgts.ndim == 2 else all_trgts
        acc = self.metric(all_preds.argmax(1), hard_targets)
        self.log("train_acc", acc, batch_size=len(all_preds))

        self.train_step_preds.clear()
        self.train_step_trgts.clear()

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.val_step_preds.append(prd)
        self.val_step_trgts.append(lab)
        self.log('val_loss', loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)
        all_trgts = torch.cat(self.val_step_trgts, dim=0)

        # Compute metrics using hard labels
        hard_targets = all_trgts.argmax(1) if all_trgts.ndim == 2 else all_trgts
        acc = self.metric(all_preds.argmax(1), hard_targets)
        self.log("val_acc", acc, batch_size=len(all_preds))

        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)        
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())


class TunnerModelFS(L.LightningModule):
    def __init__(
        self,
        model = models.resnet18(pretrained=True),
        output_dim = 512,
        lr=0.001, momentum=0.9, nesterov = True, weight_decay = 1, batch_size = 64, num_classes = 100,
    ):
        
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
                nn.Linear(output_dim, num_classes)
            )
        self.softmax = nn.Softmax(dim=1)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        if num_classes == 2:
            self.metric = BinaryAccuracy()
        else:
            self.metric = MulticlassAccuracy(num_classes=num_classes)

        self.predictions = []
        self.targets = []

        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []
        self.train_loss = []
        self.val_loss = []
        
    def forward(self, x):
        x = self.model(x)
        if isinstance(x, tuple):
            x = x[-1]
        out = self.fc(x)
        return out
    
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)
        #if self.num_classes != 40:
        #    return F.binary_cross_entropy_with_logits(yp.float(), y) 
    
    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum, nesterov = self.nesterov, weight_decay = self.weight_decay)
        optimizer = optim.AdamW(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, eta_mean = 5e-5, T_max = 100)
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler
    
    def process_batch(self, batch):
        img = batch[0].to(self.device)
        lab = batch[1].to(self.device)
        if img.shape[1] < 3:#?
            img = torch.stack((img,img,img), dim = 1).squeeze(2)
        out = self.forward(img)
        prd = torch.softmax(out, dim=1)
        loss = self.compute_loss(prd, lab)
        return loss, prd, lab

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.train_step_preds.append(prd)
        self.train_step_trgts.append(lab)
        self.log('train_loss', loss, 
            on_step=True,
            on_epoch=False,
            prog_bar=True, batch_size=self.batch_size)        
        '''batch_ratio = len(np.where(lab.cpu().numpy() == 1)[0]) / len(np.where(lab.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=self.batch_size)                        
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)'''
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)
        all_trgts = torch.cat(self.train_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        acc = self.metric(all_preds.argmax(1), all_trgts)
        self.log('train_auc', auc, batch_size=len(all_preds))
        self.log('train_acc', acc, batch_size=len(all_preds))
        self.train_step_preds.clear()
        self.train_step_trgts.clear()

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.val_step_preds.append(prd)
        self.val_step_trgts.append(lab)
        self.log('val_loss', loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)
        all_trgts = torch.cat(self.val_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        acc = self.metric(all_preds.argmax(1), all_trgts)
        self.log('val_auc', auc, batch_size=len(all_preds))
        self.log('val_acc', acc, batch_size=len(all_preds))
        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)        
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())


class CustomTransform:
    def __init__(self, processor, augmentations = {}):
        train_transforms_list = [v2.Resize((256, 256)), v2.CenterCrop((224, 224))]
        if augmentations.get("ColorJitter"):
            train_transforms_list.append(
                v2.ColorJitter(
                    brightness=augmentations["ColorJitter"].get("brightness", 0.2),
                    contrast=augmentations["ColorJitter"].get("contrast", 0.2),
                    saturation=augmentations["ColorJitter"].get("saturation", 0.2),
                    hue=augmentations["ColorJitter"].get("hue", 0.2),
                )
            )
        if augmentations.get("RandomHorizontalFlip", False):
            train_transforms_list.append(v2.RandomHorizontalFlip())
        if augmentations.get("RandomRotation"):
            train_transforms_list.append(
                v2.RandomRotation(augmentations["RandomRotation"])
            )
        if augmentations.get("GaussianBlur"):
            train_transforms_list.append(
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=augmentations["GaussianBlur"].get("kernel_size", 3))],
                    p=augmentations["GaussianBlur"].get("probability", 0.2),
                )
            )
        self.custom_transforms = v2.Compose(train_transforms_list)
        self.processor = processor

    def __call__(self, image):
        # Apply custom transforms first
        transformed_image = self.custom_transforms(image)        
        # Apply processor's preprocess
        processed_image = self.processor(transformed_image)
        
        return processed_image

class CustomTransform_:
    def __init__(self, processor, transform):
        self.custom_transforms = transform
        self.processor = processor

    def __call__(self, image):
        # Apply custom transforms first
        transformed_image = self.custom_transforms(image)        
        # Apply processor's preprocess
        processed_image = self.processor(transformed_image)
        
        return processed_image
def get_transform(augmentations = {}):
    train_transforms_list = [v2.Resize((256, 256)), v2.CenterCrop((224, 224))]
    '''train_transforms = v2.Compose([
        v2.Resize((256,256)),
        v2.CenterCrop((224,224)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(30),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])'''
    if augmentations.get("ColorJitter"):
        train_transforms_list.append(
            v2.ColorJitter(
                brightness=augmentations["ColorJitter"].get("brightness", 0.2),
                contrast=augmentations["ColorJitter"].get("contrast", 0.2),
                saturation=augmentations["ColorJitter"].get("saturation", 0.2),
                hue=augmentations["ColorJitter"].get("hue", 0.2),
            )
        )
    if augmentations.get("RandomHorizontalFlip", False):
        train_transforms_list.append(v2.RandomHorizontalFlip())
    if augmentations.get("RandomRotation"):
        train_transforms_list.append(
            v2.RandomRotation(augmentations["RandomRotation"])
        )
    if augmentations.get("GaussianBlur"):
        train_transforms_list.append(
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=augmentations["GaussianBlur"].get("kernel_size", 3))],
                p=augmentations["GaussianBlur"].get("probability", 0.2),
            )
        )
    # Finalize with ToTensor and Normalize
    train_transforms_list += [
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    train_transforms = v2.Compose(train_transforms_list)
    return train_transforms

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

class CIFAR100DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True,transform = None):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((224,224)),v2.ToTensor(),])
        if transform:
            self.transform = transform
        self.training_data = torchvision.datasets.CIFAR100(root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = torchvision.datasets.CIFAR100(root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class CIFAR10DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True,transform = None, augmentations={}):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([
                v2.Resize((256,256)),
                v2.CenterCrop((224,224)),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                v2.RandomRotation(15),
                v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.test_transform = v2.Compose([
            v2.Resize((224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if transform:
            ctransform = v2.Compose([
                v2.Resize((256,256)),
                v2.CenterCrop((224,224)),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                v2.RandomRotation(15),
                v2.RandomHorizontalFlip(),])
            self.transform = CustomTransform_(transform, ctransform)
            self.test_transform = transform
        self.training_data = torchvision.datasets.CIFAR10(root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = torchvision.datasets.CIFAR10(root = "./Data", train = False, download = download, transform=self.test_transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
              


class STL10DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None, augmentations={}):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((224,224)),
                                     v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.test_transform = v2.Compose([
            v2.Resize((224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if transform:
            self.transform = transform
            self.test_transform = transform
        self.training_data = torchvision.datasets.STL10(root = "./Data", split = 'train', download = download, transform=self.transform)
        self.test_data = torchvision.datasets.STL10(root = "./Data", split = 'test', download = download, transform=self.test_transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)


class SVHNDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None, augmentations={}):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([
                v2.Resize((256,256)),
                v2.CenterCrop((224,224)),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                v2.RandomRotation(15),
                #v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.test_transform = v2.Compose([
            v2.Resize((224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if transform:
            ctransform = v2.Compose([
                v2.Resize((256,256)),
                v2.CenterCrop((224,224)),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                v2.RandomRotation(15),])
            self.transform = CustomTransform_(transform, ctransform)
            self.test_transform = transform
        self.training_data = torchvision.datasets.SVHN(root = "./Data", split = 'train', download = download, transform=self.transform)
        self.test_data = torchvision.datasets.SVHN(root = "./Data", split = 'test', download = download, transform=self.test_transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)

class DTDDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None, augmentations={}):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((224,224)),v2.ToTensor(),
                                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        train_transforms = get_transform(augmentations)
        if transform:
            self.transform = transform
            train_transforms = CustomTransform(transform, augmentations)
        self.training_data = torchvision.datasets.DTD(root = "./Data", split = "train", download = download, transform=train_transforms)
        self.test_data = torchvision.datasets.DTD(root = "./Data", split = "test", download = download, transform=self.transform)
        self.valid_data = torchvision.datasets.DTD(root = "./Data", split = "val", download = download, transform=self.transform)


from torch.utils.data import default_collate
def collate_fn(batch):
    mixup = v2.MixUp(num_classes=100, alpha = 0.2)  # Ensure num_classes matches your dataset
    inputs, targets = default_collate(batch)  # Batch of images and labels
    inputs, targets = mixup((inputs, targets))
    hard_targets = torch.argmax(targets, dim=1)
    return inputs, targets

class FGVCDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None, augmentations={}, mixup = False):
        super().__init__(batch_size, num_workers)
        
        
        #check
        self.transform = v2.Compose([v2.Resize((224,224)), #for CNN based that are trained on ImageNet
                                     v2.ToTensor(),
                                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                       ])
        train_transforms = get_transform(augmentations)
        if transform:
            self.transform = transform
            train_transforms = CustomTransform(transform, augmentations)
        self.mixup = mixup
        self.training_data = torchvision.datasets.FGVCAircraft(root = "./data", split = "train", download = download, transform=train_transforms)
        self.test_data = torchvision.datasets.FGVCAircraft(root = "./data", split = "test", download = download, transform=self.transform)
        self.valid_data = torchvision.datasets.FGVCAircraft(root = "./data", split = "val", download = download, transform=self.transform)#annotation_level = 'manufacturer', 
    
    def train_dataloader(self):
        if self.mixup:
            return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g , collate_fn=collate_fn)
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )

    def val_dataloader(self):
        if self.mixup:
            return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )

    def test_dataloader(self):
        if self.mixup:
            return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, worker_init_fn = seed_worker, generator = g )

class CUBDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None, augmentations={}):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((224,224)),v2.ToTensor(),
                                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        train_transforms = get_transform(augmentations)
        if transform:
            self.transform = transform
            train_transforms = CustomTransform(transform, augmentations)
        self.training_data = get_cub_train(root = "./Data", train = True, download = download, transform=train_transforms)
        self.test_data = get_cub_train(root = "./Data", train = False, download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)


class Food101DataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None, augmentations={}):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((224,224)),v2.ToTensor(),
                                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        train_transforms = get_transform(augmentations)
        if transform:
            self.transform = transform
            train_transforms = CustomTransform(transform, augmentations)
        self.training_data = torchvision.datasets.Food101(root = "/export/datasets/public/Food101", split = "train", download = download, transform=self.transform)
        self.test_data = torchvision.datasets.Food101(root = "/export/datasets/public/Food101", split = "test", download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)

class CarsDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = False, transform = None, augmentations={}):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((224,224)),v2.ToTensor(),
                                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        train_transforms = get_transform(augmentations)
        if transform:
            self.transform = transform
            train_transforms = CustomTransform(transform, augmentations)
        self.training_data = torchvision.datasets.StanfordCars(root = "/export/datasets/public/", split = "train", download = download, transform=self.transform)
        self.test_data = torchvision.datasets.StanfordCars(root = "/export/datasets/public/", split = "test", download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)

class PetsDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = False, transform = None, augmentations={}):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((224,224)),v2.ToTensor(),
                                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        train_transforms = get_transform(augmentations)
        if transform:
            self.transform = transform
            train_transforms = CustomTransform(transform, augmentations)
        self.training_data = torchvision.datasets.OxfordIIITPet(root = "/export/datasets/public/Pets/", download = download, transform=self.transform)
        self.test_data = torchvision.datasets.OxfordIIITPet(root = "/export/datasets/public/Pets/", split = "test", download = download, transform=self.transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)


class PCAMDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = False):
        super().__init__(batch_size, num_workers)
        _,_, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        self.transform = v2.Compose([
                self.preprocess_val.transforms[0],
                self.preprocess_val.transforms[1],
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                self.preprocess_val.transforms[2],
                self.preprocess_val.transforms[3],
                self.preprocess_val.transforms[4],])
        self.training_data = datasets.PCAM(root = "./Data", split = 'train', download = download, transform=self.transform)
        self.valid_data = datasets.PCAM(root = "./Data", split = "val", download = download, transform=self.preprocess_val)
        self.test_data = datasets.PCAM(root = "./Data", split = "test", download = download, transform=self.preprocess_val)


class ModelImageTransform:
    def __init__(self, model_name):

        self.transforms = {
            "SegFormer": AutoImageProcessor.from_pretrained("nvidia/mit-b5", return_dict=False),
            "ViT": ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', return_dict=False),
            "Swin": AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224", return_dict=False),
            "DINOv2": AutoImageProcessor.from_pretrained('facebook/dinov2-base', return_dict=False),
            "BEiT": BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224', return_dict=False),
            "PVTv2": AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0", return_dict=False),
        }
        
        self.processor = self.transforms[model_name]
    
    def __call__(self, image):

        processed_image = self.processor(images=image, return_tensors="pt")
        return processed_image["pixel_values"].squeeze(0)
        

class QMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([
            v2.RandomRotation(15),
            v2.Resize((224,224)),v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),
        ])
        self.test_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),
        ])
        
        if transform:
            self.transform = transform
            self.test_transform = transform
        self.training_data = torchvision.datasets.QMNIST(root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = torchvision.datasets.QMNIST(root = "./Data", train = False, download = download, transform=self.test_transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class KMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([
            v2.RandomRotation(15),
            v2.Resize((224,224)),v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),
        ])
        self.test_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),
        ])
        if transform:
            self.transform = transform
            self.test_transform = transform
        self.transform_test = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = torchvision.datasets.KMNIST(root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = torchvision.datasets.KMNIST(root = "./Data", train = False, download = download, transform=self.test_transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)

class FashionMNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([
            v2.Resize((224,224)),
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),])
        self.test_transform = v2.Compose([
            v2.Resize((224,224)),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),
        ])
        if transform:
            self.transform = transform
            self.test_transform = transform
        self.training_data = torchvision.datasets.FashionMNIST(root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = torchvision.datasets.FashionMNIST(root = "./Data", train = False, download = download, transform=self.test_transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
        

class MNISTDataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, transform = None):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([v2.Resize((224,224)),
            v2.RandomRotation(15),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),])
        self.test_transform = v2.Compose([
            v2.Resize((224,224)),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),
        ])
        if transform:
            self.transform = transform
            self.test_transform = transform
        self.training_data = torchvision.datasets.MNIST(root = "./Data", train = True, download = download, transform=self.transform)
        self.test_data = torchvision.datasets.MNIST(root = "./Data", train = False, download = download, transform=self.test_transform)
        self.training_data, self.valid_data = torch.utils.data.random_split(self.training_data, [0.8, 0.2], generator=g)
                
class CustomCelebA(torchvision.datasets.CelebA):
    def __init__(self, root, split="train", target_type="attr", transform=None, target_transform=None, download=False, specific_index=None):
        # Initialize the parent CelebA class with the standard arguments
        super().__init__(root=root, split=split, target_type=target_type, transform=transform, target_transform=target_transform, download=download)
        
        self.index = specific_index
        
    def __getitem__(self, index):
        # Retrieve image and label
        image, labels = super().__getitem__(index)
        # Return image, index, and label corresponding to that index
        return image, labels[self.index]
        
class CelebADataModule(DataModule):
    def __init__(self, batch_size, num_workers, download = True, index = 20):
        super().__init__(batch_size, num_workers)
        self.transform = v2.Compose([
        	v2.RandomHorizontalFlip(p=0.5),
                #v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                v2.RandomResizedCrop((225,225), scale=(0.3, 1.0)),
                v2.ToTensor(),])
        self.transform_test = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
        self.training_data = CustomCelebA(root = "./Data", split = 'train', download = download, transform=self.transform, specific_index = index)
        self.test_data = CustomCelebA(root = "./Data", split = 'test', download = download, transform=self.transform_test, specific_index = index)
        self.valid_data = CustomCelebA(root = "./Data", split = 'valid', download = download, transform=self.transform_test, specific_index = index)
        

data_module_dict = {#"CIFAR10", "FMNIST", "MNIST", "STL10", "SVHN", "QMNIST", "KMNIST"
            "CIFAR10" : CIFAR10DataModule,
            "CIFAR100": CIFAR100DataModule,
            "FMNIST": FashionMNISTDataModule,
            "MNIST": MNISTDataModule,
            "STL10": STL10DataModule,
            "CalebA": CelebADataModule,
            "SVHN": SVHNDataModule,
            "QMNIST": QMNISTDataModule,
            "KMNIST": KMNISTDataModule,
            "Food101": Food101DataModule,
            "DTD": DTDDataModule,
            "FGVCAircraft": FGVCDataModule,
            "cub": CUBDataModule,
            "cars": CarsDataModule,
            "pets": PetsDataModule}
data_class_dict = {
            "CIFAR10" : 10,
            "CIFAR100": 100,
            "FMNIST": 10,
            "MNIST": 10,
            "STL10": 10,
            "CalebA": 2,
            "SVHN": 10,
            "QMNIST": 10,
            "KMNIST": 10,
            "PCAM": 2,
            "DTD": 47,
            "FGVCAircraft": 100,
            "cub": 200,
            "cars": 196,
            "Food101": 101,
            "pets": 37,
            }