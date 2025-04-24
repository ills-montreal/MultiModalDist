import torch
from utils.collate import make_aligned_collate_fn
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import v2
from multiDist.utils.vision_utils import VisionDataset, get_trasform_vision
from multiDist.utils.text_utils import TextDataset
from multiDist.utils.collate import *
from torch.utils.data import DataLoader
from utils.embedder_info import teachers_size



def get_dataset(mod, args):
    #!UPDATE!# Add molecule and text dataset
    if mod == "vision":
        emb = []
        for teacher in args.vision_embedders_to_simulate:
            emb.append(teachers_size[teacher])
        return VisionDataset(teachers_path=args.embeddings_dir + "/vision", list_teachers=args.vision_embedders_to_simulate, transform = get_trasform_vision(args)), emb
    elif mod == "text":
        emb = []
        for teacher in args.text_embedders_to_simulate:
            emb.append(teachers_size[teacher])
        return VisionDataset(teachers_path=args.embeddings_dir + "/vision", list_teachers=args.vision_embedders_to_simulate, transform = get_trasform_vision(args)), emb
    else:
        emb = []
        for teacher in args.molecular_embedders_to_simulate:
            emb.append(teachers_size[teacher])
        return VisionDataset(teachers_path=args.embeddings_dir + "/vision", list_teachers=args.vision_embedders_to_simulate, transform = get_trasform_vision(args)), emb

class CombinedDataset(Dataset):
    def __init__(self, datasets):
        """
        Args:
            datasets (Dict): Dictionary of Datasets for each modality.
        A combined dataset for all modalities, that for an index returns 
            x: the data, which is a text or image or molecule (depending on the index)
            embeddings: embedding of each teacher for that data
            indexes: indexes of teachers that have the embedding of x (this was used for text)
            mod: the modality of the index
        """
        self.datasets = datasets
        index = 0
        self.indices = []
        self.modalities = []
        '''Mark the index of begginging of each modality'''
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
        '''get the index of the modality and the local index in that modality'''
        mod_index = np.argmax(self.indices > idx) - 1
        mod = self.modalities[mod_index]
        ds_index = self.indices[mod_index]
        local_index = idx - ds_index
        x, embeddings, indexes = self.datasets[mod][local_index]
        return x, embeddings, indexes, mod
    
def get_embedding_loader(args):
    datasets = {}
    embd_size = {}
    '''get the datasets for each modality and the size of the embeddings'''
    for mod in args.modalities_to_simulate:
        data, embd = get_dataset(mod, args)
        datasets[mod] = data
        embd_size.setdefault(mod, []).extend(embd)

    #get the combined dataset, and split it into train and valid
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