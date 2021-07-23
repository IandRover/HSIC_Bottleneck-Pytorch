import torch
from torch import nn

from torchvision import datasets, transforms

### DATA ### 

#To download MNIST
#!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
#!tar -zxvf MNIST.tar.gz

def load_data(args, download=False):
    
    if args.dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(),])
        trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
        
    if args.dataset == "cifar":
        transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, 
                                               drop_last=True, num_workers=4, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, 
                                              drop_last=True, num_workers=4, pin_memory=False)
    return train_loader, test_loader