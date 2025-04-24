from torch.utils.data import Dataset, DataLoader

import numpy as np

teacher_embeddings = [100]

class TextDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 1000

    '''Return text, teacher embeddings and index of teachers that have the embedding'''
    def __getitem__(self, idx): 
        return "test text", [np.zeros(length) for length in teacher_embeddings], list(range(len(teacher_embeddings)))