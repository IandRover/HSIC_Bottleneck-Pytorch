import torch
from torch import nn
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F

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
        
# by lateral inhibition: a feature inhibits laterally long-range same feature
def draw_with_contraint(r, d, c):
    """r: range; d: distance, c:counts"""
    buffer = list()
    for i in range(c*100):
        a, b, e, f = np.random.randint(0, r), np.random.randint(0, r), np.random.randint(0, r), np.random.randint(0, r)
        if np.linalg.norm((a-b,e-f)) >= d:
            buffer.append([a,b,e,f])
            if len(buffer) >= c:
                return np.array(buffer)
def spatial_contrast(features, args):
    bs, cs, ws, hs = features.size()
    n_samples = 10
    coordinates = draw_with_contraint(ws, np.sqrt(ws), n_samples)
    x = features[:,:,coordinates[:,0],coordinates[:,1]]
    x = x.permute(1,0,2).reshape((cs, bs*n_samples)) # [cs, bs*n_sample]
    y = features[:,:,coordinates[:,2],coordinates[:,3]]
    y = y.permute(1,0,2).reshape((cs, bs*n_samples)) # [cs, bs*n_sample]
    x = F.normalize(x, p=2, dim=0)
    y = F.normalize(y, p=2, dim=0)
    if args.Latinb_type == "n": return torch.mean(torch.trace(torch.mm(torch.t(x), y)))
    elif args.Latinb_type == "f": return torch.mean(torch.trace(torch.mm(y, torch.t(x))))