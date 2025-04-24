import torch
import torch.nn as nn
from utils.embedder_info import get_embedder
import numpy as np 
class StudentModel(nn.Module):
    def __init__(self, embedders, embedders_size, dim, mods, device):
        """
        Args:
            embedders (list): A list of embedding layers.
            dim (int): The output dimension for the whole model.
        """
        super(StudentModel, self).__init__()
        
        self.embedders = nn.ModuleList(embedders) 
        self.fcs = nn.ModuleList() 
        self.mods = mods
        
        for emb_dim in embedders_size:
            '''map the output of the embedder to the desired dimension for each modalitites' embedder'''
            #embedder_output_dim = [layer.out_features for layer in embedder.modules() if isinstance(layer, nn.Linear)][-1] 
            self.fcs.append(nn.Linear(emb_dim, dim))
        '''pass through the common layer, which is then given to the knifes'''
        self.F = nn.Linear(dim, dim)
        self.dim = dim
        self.device = device
    
    def forward(self, batch):
        out = torch.zeros(len(batch[self.mods[0]]),self.dim).to(self.device)
        for index, mod in enumerate(self.mods):
            '''get data of each modality in the batch'''
            indexes = [i for i in range(len(batch[mod + "_emb"])) if batch[mod + "_emb"][i] is not None]
            data = [batch[mod][i] for i in range(len(batch[mod + "_emb"])) if batch[mod + "_emb"][i] is not None]
            '''pass data of each modality through the corresponding embedder and fc layer'''
            x = self.embedders[index](torch.tensor(data).to(self.device))
            if isinstance(x, tuple):
                x = x[0]
            x = self.fcs[index](x)
            '''pass the mapped embeddings through the common layer'''
            x = self.F(x)
            out[indexes] = x
        return out





def get_student_model(args):
    embedders = []
    embedders_size = []
    args_dict = vars(args)
    for mod in args.modalities_to_simulate:
        embedders.append(get_embedder(args_dict[mod + "_student"]).to(args.device))
        embedders_size.append(args.student_emb_size[mod])
    del args_dict

    model = StudentModel(embedders, embedders_size, args.out_dim, args.modalities_to_simulate, args.device)
    model = model.to(args.device)
    return model