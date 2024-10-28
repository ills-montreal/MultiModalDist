import torch
import torch.nn as nn
from Embeddings.vision.teachers_dict import teachers_dict
import numpy as np 
class StudentModel(nn.Module):
    def __init__(self, embedders, dim, mods, device):
        """
        Args:
            embedders (list): A list of embedding layers.
            dim (int): The output dimension for the whole model.
        """
        super(StudentModel, self).__init__()
        
        self.embedders = nn.ModuleList(embedders) 
        self.fcs = nn.ModuleList() 
        self.mods = mods
        
        for embedder in self.embedders:
            embedder_output_dim = [layer.out_features for layer in embedder.modules() if isinstance(layer, nn.Linear)][-1] 
            self.fcs.append(nn.Linear(embedder_output_dim, dim))
        
        self.F = nn.Linear(dim, dim)
        self.dim = dim
        self.device = device
    
    def forward(self, batch):
        out = torch.zeros(len(batch[self.mods[0]]),self.dim).to(self.device)
        for index, mod in enumerate(self.mods):
            indexes = [i for i in range(len(batch[mod + "_emb"])) if batch[mod + "_emb"][i] is not None]
            data = [batch[mod][i] for i in range(len(batch[mod + "_emb"])) if batch[mod + "_emb"][i] is not None]
            x = self.embedders[index](torch.tensor(data).to(self.device))
            x = self.fcs[index](x)
            x = self.F(x)
            out[indexes] = x
            
        return out





def get_student_model(args):
    embedders = []
    if "text" in args.modalities_to_simulate:
        embedders.append(teachers_dict[args.vision_student](pretrained=True).to(args.device))
    if "vision" in args.modalities_to_simulate:
        embedders.append(teachers_dict[args.vision_student](pretrained=True).to(args.device))

    #if "molecular" in args.modalities_to_simulate:

    model = StudentModel(embedders, args.dim, args.modalities_to_simulate, args.device)
    model = model.to(args.device)
    return model