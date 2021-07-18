import torch
from torch import nn
from torchvision import datasets, transforms

### DATA ###
def load_data(batch_size, download=False):
    
#     if download:
#         !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
#         !tar -zxvf MNIST.tar.gz
    
    batch_size = batch_size
    transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean=(0.5,), std=(0.5,))
                                   ])
    X_train = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, 
                                               drop_last=True, num_workers=4, pin_memory=False)
    X_test = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False, 
                                              drop_last=True, num_workers=4, pin_memory=False)
    return train_loader, test_loader