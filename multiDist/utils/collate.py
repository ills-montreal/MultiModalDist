import numpy as np
from types import FunctionType
from typing import List, Tuple, Dict, Optional, Callable
def make_aligned_collate_fn(modalities, modality_collate_fn):
    def collate_fn(batch):
        input, embedding, teacher_idxs, data_type = list(zip(*batch))
        '''returns only modality specific data, with None for other modalities to prevent error'''
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
            indexes = [i for i in range(len(modaity_especific_inputs[mod])) if modaity_especific_inputs[mod][i] is not None]
            if mod in modality_collate_fn.keys() and mod in data_type:
                batch = list(zip([modaity_especific_inputs[mod][i] for i in indexes] \
                                , [modaity_especific_embeddings[mod][i] for i in indexes], \
                                    [modaity_especific_teacher_idxs[mod][i] for i in indexes]))
                modaity_especific_inputs_, modaity_especific_embeddings_, modaity_especific_teacher_idxs_ = modality_collate_fn[mod](batch)
                
                for i , inp in enumerate(modaity_especific_inputs_):
                    modaity_especific_inputs[mod][indexes[i]] = inp
                    modaity_especific_embeddings[mod][indexes[i]] = [emb[i] for emb in modaity_especific_embeddings_]
                    modaity_especific_teacher_idxs[mod][indexes[i]] = [idx[i] for idx in modaity_especific_teacher_idxs_]
            batch_dict[mod] = modaity_especific_inputs[mod]
            batch_dict[mod+"_emb"] = modaity_especific_embeddings[mod] # bach_size x num_teachers x emb_dim
            batch_dict[mod+"_indexes"] = modaity_especific_teacher_idxs[mod]
        return batch_dict
    return collate_fn

