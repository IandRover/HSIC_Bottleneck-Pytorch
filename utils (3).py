import torch
from torchvision import datasets, transforms

def kernel_matrix(x : torch.Tensor, sigma):
    dim = len(x.size())
    x1  = torch.unsqueeze(x, 0)
    x2  = torch.unsqueeze(x, 1)
    axis= tuple(range(2, dim+1))
    if dim > 1:
        return torch.exp( -0.5 * torch.sum(torch.pow(x1-x2, 2), axis=axis) / sigma**2)
    else:
        return torch.exp( -0.5 * torch.pow(x1-x2, 2) / sigma**2)

def kernel_student(x : torch.Tensor, sigma):
    dim = len(x.size())
    m = x.size(0)
    x1  = torch.unsqueeze(x, 0)
    x2  = torch.unsqueeze(x, 1)
    axis= tuple(range(2, dim+1))
    return 1/(sigma + torch.mean(torch.pow(x1-x2, 2), axis=axis))*sigma
    
def HSIC(Kx, Ky, m):
    xy = torch.matmul(Kx, Ky)
    h  = torch.trace(xy) / m**2 + torch.mean(Kx)*torch.mean(Ky) - \
        2 * torch.mean(xy)/m
    return h*(m/(m-1))**2 

def mean(K):
    return K - torch.mean(K, 1, keepdim=True)
def norm_HSIC(Kx, Ky, m, device):
    Kxc = mean(Kx)
    Kyc = mean(Ky)
    Kxi = torch.inverse(Kxc + 1e-5 * m * torch.eye(m).to(device))
    Kyi = torch.inverse(Kyc + 1e-5 * m * torch.eye(m).to(device))
    Rx = (Kxc.mm(Kxi))
    Ry = (Kyc.mm(Kyi))
    Pxy = torch.mean(torch.mul(Rx, Ry.t()))
    return Pxy

    

def load_data(batch_size, download=False):
    
    if download:
        new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
        datasets.MNIST.resources = [
           ('/'.join([new_mirror, url.split('/')[-1]]), md5)
           for url, md5 in datasets.MNIST.resources]
        train_dataset = datasets.MNIST("./data", train=True, download=True, transforms=transforms.Compose([transforms.ToTensor()]))
    
    batch_size = batch_size
    transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean=(0.5,), std=(0.5,))
                                   ])
    X_train = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, 
                                               drop_last=True, num_workers=2, pin_memory=True)
    X_test = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False, 
                                              drop_last=True, num_workers=2, pin_memory=True)
    return train_loader, test_loader