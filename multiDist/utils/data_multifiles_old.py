import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
from torchvision.transforms import v2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import torchvision
from datasets import load_dataset
import os
from functools import partial
from torch.utils.data import DataLoader

class TeacherEmbeddingDataset(Dataset):
    def __init__(self, teacher_paths, teacher_name, datasets, length, indices):
        self.teacher_paths = teacher_paths
        self.teacher_name = teacher_name
        self.length = length
        self.indices = indices

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        index = np.argmax(self.indices > idx) - 1
        ds_index = self.indices[index]
        local_index = idx - ds_index
        data = np.load(self.embeding_index_to_file[ds_index], mmap_mode="r")
        return data[local_index]
        

        
vision_datasets_dict = {
            "CIFAR10" : torchvision.datasets.CIFAR10,
            "FMNIST": torchvision.datasets.FashionMNIST,
}


def worker_init_factory(idx):
    def worker_init_fn(worker_id, idx):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if idx is None:
            idx = list(range(len(dataset)))
        n_data = len(idx)
        n_data_per_worker = [0] + [
            (1 + k) * (n_data // worker_info.num_workers)
            + (k == 0) * (n_data % worker_info.num_workers)
            for k in range(worker_info.num_workers)
        ]
        idx_split = idx[n_data_per_worker[worker_id] : n_data_per_worker[worker_id + 1]]
        dataset.update_idx(idx_split)

    return partial(worker_init_fn, idx=idx)

class MultiTeacherAlignedEmbeddingDataset:

    def __init__(  self,  teachers_path = "./Embeddings", modalities = ["text", "vision"], embedders_to_simulate = {"text": ["gte"], "vision": ["mnasnet", "shufflenet"]}):
        self.teacher_paths = teachers_path
        self.modalities = modalities
        index = 0
        self.embeding_index_to_ds = {}
        self.embeding_index_to_ds_name = {}
        self.embedding_sizes = {}
        for mod in modalities:
            for embd in embedders_to_simulate[mod]:
                embds_pth = glob.glob(teachers_path + "/" + mod  + "/"+ embd +"_*.npy" )
                self.embedding_sizes[embd] = np.load(embds_pth[0]).shape[1]
                for pth in embds_pth:
                    dataset = pth.split(".npy")[0].split(embd)[1].replace("_","")
                    if mod == "vision":
                        transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])
                        data = vision_datasets_dict[dataset](root = "./Data",download = True, transform=transform)
                    elif mod == "text":
                        data = load_dataset(dataset)['train']
                        data = data.remove_columns('idx')
                        data = [(d['sentence'], d['label']) for d in data]
                    self.embeding_index_to_ds[index] = data
                    self.embeding_index_to_ds_name[index] = dataset
                    index = index + len(data)
        self.length = index
        self.teachers = []
        self.teachers_ds = []
        self.indices = np.array(list(self.embeding_index_to_ds.keys()))
        self.embedders_to_simulate = embedders_to_simulate
        
    
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        index = np.argmax(self.indices > idx) - 1
        ds_index = self.indices[index]
        local_index = idx - ds_index
        x, lable = self.embeding_index_to_ds[ds_index][local_index]
        if not isinstance(x, str):
            x = np.array(x)
            if x.shape[0] < 3: #if 2D image
                x = torch.stack((x,x,x), dim = 0).squeeze(1)
        emb = []
        for mod in self.modalities:
            for embd in self.embedders_to_simulate[mod]:
                file_path = self.teacher_paths + "/" + mod  + "/"+ embd +"_" + self.embeding_index_to_ds_name[ds_index] + ".npy"
                if os.path.exists(file_path):
                    data = np.load(file_path, mmap_mode="r")
                    emb.append(data[local_index])
                else:
                    emb.append(None)
        return x, emb

def get_embedding_loader(args):
    dataset = MultiTeacherAlignedEmbeddingDataset()
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        drop_last=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )

    return train_loader, valid_loader, dataset.embedding_sizes, {"train": len(train_dataset), "valid": len(valid_dataset)}
