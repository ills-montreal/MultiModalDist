import torchvision.models as models
teachers_dict = {
    #"resnet18" : models.resnet18,
    #"alexnet" : models.alexnet,
    #"squeezenet" : models.squeezenet1_0,
    #"vgg16" : models.vgg16,
    #"densenet" : models.densenet161,
    #"inception" : models.inception_v3,#RuntimeError: Calculated padded input size per channel: (3 x 3). Kernel size: (5 x 5). Kernel size can't be greater than actual input size
    #"googlenet" : models.googlenet,
    "shufflenet" : models.shufflenet_v2_x1_0,
    #"mobilenet" : models.mobilenet_v2,
    #"resnext50_32x4d" : models.resnext50_32x4d,
    #"wide_resnet50_2" : models.wide_resnet50_2,
    "mnasnet" : models.mnasnet1_0,
    }

