import torch
from torch import nn
from torchvision import datasets, transforms

### MODEL ###
class MLPBlock(nn.Module):
    def __init__(self, inplane, outplane, affine=False):
        super(MLPBlock, self).__init__()
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
        if args.dataset == "mnist":
            self.units = [784, 256, 128, 128]
            self.output_layer  = nn.Linear(self.units[-1], 10)        
        elif args.dataset == "cifar":
            self.units = [3072, 256, 256, 256, 256, 256]
            self.output_layer  = nn.Linear(self.units[-1], 10)

        self.module_list = nn.ModuleList( [MLPBlock(self.units[i], self.units[i+1], affine=self.bn_affine) for i in range(len(self.units)-1)])
        self.f3 = nn.Dropout(p=0.2)
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
    
    
### MODEL ###
# class ConvBlock(nn.Module):
#     def __init__(self, inplane, outplane, affine=False):
#         super(ConvBlock, self).__init__()
#         self.linear = nn.Conv2d(inplane, outplane, 3, stride=2, padding=1)
#         self.bn = nn.BatchNorm2d(outplane, affine=affine)
#         self.act = nn.GELU()
#     def forward(self, x):
#         x = self.linear(x)
#         x = self.act(x)
#         x = self.bn(x)
#         return x
    
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane, affine=False):
        super(ConvBlock, self).__init__()
        self.linear = nn.Conv2d(inplane, outplane, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(outplane, affine=affine)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.bn(x)
        return x
        
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.bn_affine = True if args.bn_affine == 1 else False
        if args.dataset == "mnist":
            self.units = [1, 32, 64, 64, 64]
            self.output_layer  = nn.Linear(self.units[-1], 10)        
        elif args.dataset == "cifar":
            self.units = [3, 32, 64, 64, 64]
            self.bn = nn.BatchNorm1d(self.units[-1]*4, affine=args.bn_affine)
            self.output_layer  = nn.Linear(self.units[-1]*4, 10)
            self.size = (args.batchsize, 3, 32, 32)
        self.module_list = nn.ModuleList( [ConvBlock(self.units[i], self.units[i+1], affine=self.bn_affine) for i in range(len(self.units)-1)])
                
        self.f3 = nn.Dropout(p=0.2)
        self.act2 = nn.ReLU()
        self.AP = torch.nn.AvgPool2d(2, stride=1)
        
    def forward(self, data):
        x = data.view(self.size)
        output = []
        for module in self.module_list:
            x_ = module(x.detach())
            x = module(x)
            output.append(x_)
#         x = self.AP(x).view(self.size[0], -1)
        x = torch.flatten(x, 1)
#         print(x.shape)
        x = self.f3(x)
        x_ = self.act2(self.output_layer(x.detach()))
        x = self.act2(self.output_layer(x))
        output.append(x_)
        return x, output