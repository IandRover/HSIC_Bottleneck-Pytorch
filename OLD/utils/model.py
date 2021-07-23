import torch
from torch import nn
from torchvision import datasets, transforms

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