import numpy as np
from types import FunctionType
from typing import List, Tuple, Dict, Optional, Callable
def make_aligned_collate_fn(modalities, modality_collate_fn):
    def collate_fn(batch):
        input, embedding, teacher_idxs, data_type = list(zip(*batch))
        modaity_especific_inputs = {mod: [None] * len(input) for mod in modalities}
        modaity_especific_embeddings = {mod: [None] * len(input) for mod in modalities}
        modaity_especific_teacher_idxs = {mod: [None] * len(input) for mod in modalities}
        for i in range(len(input)):
            mod = data_type[i]
            modaity_especific_inputs[mod][i]=input[i]
            modaity_especific_embeddings[mod][i]=embedding[i]
            modaity_especific_teacher_idxs[mod][i]=teacher_idxs[i]
        batch_dict = {}
        for mod in modalities:
            if mod in modality_collate_fn.keys() and mod in data_type:
                batch = list(zip(modaity_especific_inputs[mod], modaity_especific_embeddings[mod], modaity_especific_teacher_idxs[mod]))
                batch = modality_collate_fn[mod](batch)
                modaity_especific_inputs[mod], modaity_especific_embeddings[mod], modaity_especific_teacher_idxs[mod] = list(zip(*batch))
            batch_dict[mod] = modaity_especific_inputs[mod]
            batch_dict[mod+"_emb"] = modaity_especific_embeddings[mod]
            batch_dict[mod+"_indexes"] = modaity_especific_teacher_idxs[mod]
        return batch_dict
    return collate_fn

