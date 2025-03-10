import torchvision
datasets_dict = {
            "CIFAR10" : torchvision.datasets.CIFAR10,
            #"CIFAR100": torchvision.datasets.CIFAR100,
            #"FakeData": torchvision.datasets.FakeData,#a random data
            #"FMNIST": torchvision.datasets.FashionMNIST,
            #"Flickr8": torchvision.datasets.Flickr8k,#needs to be downloaded manually
            #"Flickr30": torchvision.datasets.Flickr30k,#needs to be downloaded manually
            #"ImageNet": torchvision.datasets.ImageNet,#RuntimeError: The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupted. You need to download it externally and place it in ./Data.
            #"LSUN": torchvision.datasets.LSUN,#needs to be downloaded manually
            #"MNIST": torchvision.datasets.MNIST,
            #"Places365": torchvision.datasets.Places365,too big
            #"STL10": torchvision.datasets.STL10,#(image, target) where target is index of the target class.
            #"CalebA": torchvision.datasets.CelebA,
            #"SVHN": torchvision.datasets.SVHN,#	(image, target) where target is index of the target class.
            #"QMNIST": torchvision.datasets.QMNIST,#?
            #"EMNIST": torchvision.datasets.EMNIST,#?TypeError: EMNIST.__init__() missing 1 required positional argument: 'split'
            #"KMNIST": torchvision.datasets.KMNIST,#?
            #"Omniglot": torchvision.datasets.Omniglot,#?
            #"CityScapes": torchvision.datasets.Cityscapes,
            #"COCO": torchvision.datasets.CocoCaptions,#Tuple (image, target). target is a list of captions for the image.
            #"SBU": torchvision.datasets.SBU, #image, caption
            #"Detection": torchvision.datasets.CocoDetection,
            #"PhotoTour": torchvision.datasets.PhotoTour,
            #"SBD": torchvision.datasets.SBDataset,
            #"USPS": torchvision.datasets.USPS,
            #"VOC": torchvision.datasets.VOCDetection, #(image, target) where target is a dictionary of the XML tree.
            } #test