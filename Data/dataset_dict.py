import torchvision
datasets_dict = {
            "CIFAR10" : torchvision.datasets.CIFAR10,
            "FMNIST": torchvision.datasets.FashionMNIST,
            "MNIST": torchvision.datasets.MNIST,
            "STL10": torchvision.datasets.STL10,#(image, target) where target is index of the target class.
            #"CalebA": torchvision.datasets.CelebA,
            "SVHN": torchvision.datasets.SVHN,#	(image, target) where target is index of the target class.
            "QMNIST": torchvision.datasets.QMNIST,#?
            "KMNIST": torchvision.datasets.KMNIST,#?
            #"Omniglot": torchvision.datasets.Omniglot,#?
            }