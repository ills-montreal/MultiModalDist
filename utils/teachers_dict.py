import torchvision.models as models
teachers_dict = {
    "resnet18" : models.resnet18,
    "squeezenet" : models.squeezenet1_0,
    "densenet" : models.densenet161,
    "googlenet" : models.googlenet,
    "shufflenet" : models.shufflenet_v2_x1_0,
    "mobilenet" : models.mobilenet_v2,
    "resnext50_32x4d" : models.resnext50_32x4d,
    "wide_resnet50_2" : models.wide_resnet50_2,
    "mnasnet" : models.mnasnet1_0,
    }

