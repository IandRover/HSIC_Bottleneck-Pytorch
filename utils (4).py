import torch
from torch import nn
from torchvision import datasets, transforms

class compute_kernel(nn.Module):
    def __init__(self):
        super(compute_kernel, self).__init__()
    def kernel_rbf(self, x, sigma):
        dim = len(x.size())
        x1  = torch.unsqueeze(x, 0)
        x2  = torch.unsqueeze(x, 1)
        axis= tuple(range(2, dim+1))
        if dim > 1: return torch.exp( -0.5 * torch.sum(torch.pow(x1-x2, 2), axis=axis) / sigma**2)
        else: return torch.exp( -0.5 * torch.pow(x1-x2, 2) / sigma**2)
    def kernel_student(self, x, sigma):
        dim = len(x.size())
        m = x.size(0)
        x1  = torch.unsqueeze(x, 0)
        x2  = torch.unsqueeze(x, 1)
        axis= tuple(range(2, dim+1))
        return 1/(sigma + torch.mean(torch.pow(x1-x2, 2), axis=axis))*sigma    
    def forward(self, x : torch.Tensor, sigma, kerneltype):
        if kerneltype == "rbf": return self.kernel_rbf(x, sigma)
        elif kerneltype == "student": return self.kernel_student(x, sigma)

class compute_HSIC(nn.Module):
    def __init__(self, kerneltype):
        super(compute_HSIC, self).__init__()
        self.kerneltype = kerneltype
    def mean(self, K): return K - torch.mean(K, 1, keepdim=True)
    def HSIC(self, Kx, Ky, m, device):
        xy = torch.matmul(Kx, Ky)
        h  = torch.trace(xy) / m**2 + torch.mean(Kx)*torch.mean(Ky) - 2 * torch.mean(xy)/m
        return h*(m/(m-1))**2       
    def norm_HSIC(self, Kx, Ky, m, device):
        Kxc, Kyc = self.mean(Kx), self.mean(Ky)
        Kxi = torch.inverse(Kxc + 1e-5 * m * torch.eye(m).to(device))
        Kyi = torch.inverse(Kyc + 1e-5 * m * torch.eye(m).to(device))
        Rx, Ry = (Kxc.mm(Kxi)), (Kyc.mm(Kyi))
        Pxy = torch.mean(torch.mul(Rx, Ry.t()))
        return Pxy
    def forward(self, Kx, Ky, m, device):
        if self.kerneltype == "nHSIC": return self.norm_HSIC(Kx, Ky, m, device)
        elif self.kerneltype == "HSIC": return self.norm_HSIC(Kx, Ky, m)

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
                                               drop_last=True, num_workers=4, pin_memory=True)
    X_test = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=False, 
                                              drop_last=True, num_workers=4, pin_memory=True)
    return train_loader, test_loader


### MODEL ###
class Block(nn.Module):
    def __init__(self, inplane, outplane, affine=False):
        super(Block, self).__init__()
        self.linear = nn.Linear(inplane, outplane)
        self.bn = nn.BatchNorm1d(outplane, affine=affine)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.bn(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.bn_affine = True if args.bn_affine == 1 else False
        self.units = [784, 256, 128, 128]
        self.module_list = nn.ModuleList( [Block(self.units[i], self.units[i+1], affine=self.bn_affine) for i in range(len(self.units)-1)])
        
        self.f3 = nn.Dropout(p=0.2)
        self.output_layer  = nn.Linear(self.units[-1], 10)        
        self.act2 = nn.ReLU()
        
    def forward(self, data):
        x = data
        output = []
        for module in self.module_list:
            x_ = module(x.detach())
            x = module(x)
            output.append(x_)
        x = self.f3(x)
        x_ = self.act2(self.output_layer(x.detach()))
        x = self.act2(self.output_layer(x))
        output.append(x_)
        return x, output
    
def get_filename(args):
    filename = "{}_{}{}_S{}L{}_F{}".format(args.loss, args.kernel_forward[0], args.kernel_backward[0], args.sigma_, args.lambda_, args.forward)
    
    if args.bn_affine == 1:
        filename += "_bn"
        
    return filename