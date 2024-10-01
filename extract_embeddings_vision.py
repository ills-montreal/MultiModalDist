import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

datasets = {
            "CIFAR10" : torchvision.datasets.CIFAR10,
            "FMNIST": torchvision.datasets.FashionMNIST,
            "MNIST": torchvision.datasets.MNIST,
            "STL10": torchvision.datasets.STL10,#(image, target) where target is index of the target class.
            "CalebA": torchvision.datasets.CelebA,
            "SVHN": torchvision.datasets.SVHN,#	(image, target) where target is index of the target class.
            "QMNIST": torchvision.datasets.QMNIST,#?
            "KMNIST": torchvision.datasets.KMNIST,#?
            "Omniglot": torchvision.datasets.Omniglot,#?
            }
teachers = {
    "resnet18" : models.resnet18(pretrained=True),
    "squeezenet" : models.squeezenet1_0(pretrained=True),
    "densenet" : models.densenet161(pretrained=True),
    "googlenet" : models.googlenet(pretrained=True),
    "shufflenet" : models.shufflenet_v2_x1_0(pretrained=True),
    "mobilenet" : models.mobilenet_v2(pretrained=True),
    "resnext50_32x4d" : models.resnext50_32x4d(pretrained=True),
    "wide_resnet50_2" : models.wide_resnet50_2(pretrained=True),
    "mnasnet" : models.mnasnet1_0(pretrained=True),
    }

batch_size = 64
num_workers = 10
for teacher_name in teachers.keys():
    for dataset_name in datasets.keys():
        output = "Embeddings/"+teacher_name + "_" + dataset_name
        transform = v2.Compose([v2.Resize((225,225)),v2.ToTensor(),])#?
        data = datasets[dataset_name](root = "./Data",download = True, transform=transform)
        dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        teacher = teachers[teacher_name]
        teacher.eval()
        teacher.to(device)
        emb = torch.empty((0, 1000), dtype=torch.float32)#.to(device)
        for batch in dataloader:
            img, label = batch
            if img.shape[1] < 3:
                img = torch.stack((img,img,img), dim = 1).squeeze(2)
            out = teacher(img.to(device))
            emb = torch.cat((emb, out.cpu().detach()), 0)
        print("saving" + output)
        np.save(output+'.npy', emb.cpu().detach().numpy()) 