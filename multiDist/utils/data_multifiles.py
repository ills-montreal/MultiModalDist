import torch
from utils.collate import make_aligned_collate_fn
from torch.utils.data import Dataset
import numpy as np
from Embeddings.vision.Dataset import VisionDataset
from Embeddings.text.Dataset import TextDataset
from multiDist.utils.collate import *
from torch.utils.data import DataLoader

def get_dataset(mod, args):
    if mod == "vision":
        return VisionDataset(teachers_path=args.embeddings_dir + "/vision", list_teachers=args.vision_embedders_to_simulate), [1000]*len(args.vision_embedders_to_simulate)
    if mod == "text":
        return VisionDataset(teachers_path=args.embeddings_dir + "/vision", list_teachers=args.vision_embedders_to_simulate), [1000]*len(args.vision_embedders_to_simulate)

class CombinedDataset(Dataset):
    def __init__(self, datasets):
        """
        Args:
            datasets (Dict): Dictionary of Datasets for each modality.
        """
        self.datasets = datasets
        index = 0
        self.indices = []
        self.modalities = []
        for mod in self.datasets.keys():
            self.modalities.append(mod)
            self.indices.append(index)
            index = index + len(self.datasets[mod])
        self.indices = np.array(self.indices)
        self.length = index

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index to retrieve.

        Returns:
            tuple: (data, list of vectors, indexes, mod)
                data: text or image or molecule (depending on the index)
                list of teacher embeddings: the associated teacher embeddings
                indexes: indexes of teacher with embedings of the data
                mod: modality of the index
        """
        mod_index = np.argmax(self.indices > idx) - 1
        mod = self.modalities[mod_index]
        ds_index = self.indices[mod_index]
        local_index = idx - ds_index
        x, embeddings, indexes = self.datasets[mod][local_index]
        return x, embeddings, indexes, mod
    
def get_embedding_loader(args):
    datasets = {}
    embd_size = []
    for mod in args.modalities_to_simulate:
        data, embd = get_dataset(mod, args)
        datasets[mod] = data
        embd_size.extend(embd)

    dataset = CombinedDataset(datasets)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        collate_fn=make_aligned_collate_fn(args.modalities_to_simulate, modality_collate_fn = {}),
        drop_last=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        collate_fn=make_aligned_collate_fn(args.modalities_to_simulate, modality_collate_fn = {}),
    )
    return train_loader, valid_loader, embd_size