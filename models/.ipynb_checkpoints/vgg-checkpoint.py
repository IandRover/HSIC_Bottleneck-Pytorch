import torch
from torch import nn

### MODEL ###
def FeatureBlock(cfg, affine=False):
    layers = list()
    for i in range(len(cfg)-1):
        block = list()
        if cfg[i] == "M":
            continue
        elif cfg[i+1] != "M" and i != (len(cfg)-2):
            block = [nn.Conv2d(cfg[i], cfg[i+1], 3, stride=1, padding=1),
                     nn.GELU(),
                     nn.BatchNorm2d(cfg[i+1], affine=affine),
                     ]
        elif cfg[i+1] == "M" and i != (len(cfg)-2):
            block = [nn.Conv2d(cfg[i], cfg[i+2], 3, stride=1, padding=1),
                     nn.GELU(),
                     nn.MaxPool2d(2, stride=2),
                     nn.BatchNorm2d(cfg[i+2], affine=affine)
                    ]
        else: 
            continue
        layers += [nn.Sequential(*block)]        
    return layers

def LinearBlock(cfg, affine=False):
    layers = list()
    for i in range(len(cfg)-1):
        block = list()
        if i == (len(cfg)-1):
            break
        elif i == (len(cfg)-2):
            block = [nn.Linear(cfg[i], cfg[i+1]),
                        nn.GELU()]
        else:
            block = [nn.Linear(cfg[i], cfg[i+1]),
                        nn.GELU(),
                        nn.BatchNorm1d(cfg[i+1], affine=affine)]
        if block == list(): continue
        layers += [nn.Sequential(*block)]
    return layers

# class VGG(nn.Module):
#     def __init__(self, args):
#         super(VGG, self).__init__()
        
#         cfg = {'VGG': [3, 'M', 32, 'M', 64, 'M', 64, 'M', 64, 'M'],
#                'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#                'L': [64*4, 10],
#               }
        
#         self.bn_affine = True if args.bn_affine == 1 else False
#         if args.dataset == "cifar": 
#             self.size = (args.batchsize, 3, 32, 32)
#             self.feature = nn.ModuleList(FeatureBlock(cfg["VGG"]))
#         if args.dataset == "cifar": 
#             self.size = (args.batchsize, 1, 28, 28)
#             self.feature = nn.ModuleList(FeatureBlock(cfg["VGG"]))
#         self.linear = nn.ModuleList(LinearBlock(cfg["L"]))
#         self.dropout = nn.Dropout(p=0.2)
        
#     def forward(self, data):
#         x = data.view(self.size)
#         output = []
#         for module in self.feature:
#             x_ = module(x.detach())
#             x = module(x)
#             output.append(x_)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         for module in self.linear:
#             x_ = module(x.detach())
#             x = module(x)
#             output.append(x_)
#         return x, output

    
class VGG(nn.Module):
    def __init__(self, args):
        super(VGG, self).__init__()
        
        cfg = {'VGG11_mnist': [1, 32, 'M', 32, 'M', 32, 32, 'M', 32, 32, 'M'],
               'VGG11_cifar': [3, 'M', 64, 'M', 64, 'M', 64, 64, 'M', 64, 64, 'M'],
               'L_cifar': [64*4, 10],
               'L_mnist': [32*9, 10],
              }
        
        self.bn_affine = True if args.bn_affine == 1 else False
        if args.dataset == "cifar": 
            self.size = (args.batchsize, 3, 32, 32)
            self.feature = nn.ModuleList(FeatureBlock(cfg["VGG11_cifar"]))
            self.linear = nn.ModuleList(LinearBlock(cfg["L_cifar"]))
        if args.dataset == "mnist":
            self.size = (args.batchsize, 1, 28, 28)
            self.feature = nn.ModuleList(FeatureBlock(cfg["VGG11_mnist"]))
            self.linear = nn.ModuleList(LinearBlock(cfg["L_mnist"]))
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, data):
        x = data.view(self.size)
        output = []
        for module in self.feature:
            x_ = module(x.detach())
            x = module(x)
            output.append(x_)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        for module in self.linear:
            x_ = module(x.detach())
            x = module(x)
            output.append(x_)
        return x, output

# OLD 
# class ConvBlock(nn.Module):
#     def __init__(self, dims, affine=False):
#         super(ConvBlock, self).__init__()
#         if "M" in dims:
#             self.num_conv = len(dims)-1
#         self.ins = dims[:-1]
#         self.outs = dims[1:]
        
#         layers = list()
#         for i in range(self.num_conv):
#             layers += [nn.Conv2d(self.ins[i], self.outs[i], 3, stride=1, padding=1),
#                             nn.GELU(),
#                             nn.BatchNorm2d(self.outs[i], affine=affine)]
#         layers += [nn.MaxPool2d(2, stride=2)]
#         self.layers = nn.Sequential(*layers) 
            
#     def forward(self, x):
#         return self.layers(x)
# def read_cfg(cfg):
#     temp = list()
#     tempa = list()
#     for i in range(len(cfg)-1):        
#         if cfg[i+1] == 'M':
#             temp.append([cfg[i], "M"])
#             tempa = list()
#         else:
#             temp.append([cfg[i]])
#             tempa = list()
#     return temp
# class VGG(nn.Module):
#     def __init__(self, args):
#         super(VGG, self).__init__()
#         self.bn_affine = True if args.bn_affine == 1 else False
              
#         if args.dataset == "cifar":
#             self.units = read_cfg(cfg["VGG11"])
#             self.output_layer  = nn.Linear(cfg["VGG11"][-1][-1], 10)
#             self.size = (args.batchsize, 3, 32, 32)
            
#         self.module_list = nn.ModuleList( [ConvBlock(unit, affine=self.bn_affine) for unit in (self.units)])
#         self.module_list = nn.ModuleList(FeatureBlock(cfg["VGG11"]))
                
#         self.f3 = nn.Dropout(p=0.2)
#         self.act2 = nn.ReLU()
#         self.AP = torch.nn.AvgPool2d(2, stride=1)
        
#     def forward(self, data):
#         x = data.view(self.size)
#         output = []
#         for module in self.module_list:
#             x_ = module(x.detach())
#             x = module(x)
#             output.append(x_)
#         x = torch.flatten(x, 1)
#         x = self.f3(x)
#         x_ = self.act2(self.output_layer(x.detach()))
#         x = self.act2(self.output_layer(x))
#         output.append(x_)
#         return x, output

# class VGG_detach(nn.Module):
#     def __init__(self, args):
#         super(VGG_detach, self).__init__()
        
#         cfg = {'VGG': [3, 'M', 32, 'M', 64, 'M', 64, 'M', 64, 'M'],
#                'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#                'L': [64*4, 10],
#               }
        
#         self.bn_affine = True if args.bn_affine == 1 else False
#         if args.dataset == "cifar": self.size = (args.batchsize, 3, 32, 32)            
#         self.feature = nn.ModuleList(FeatureBlock(cfg["VGG"]))
#         self.linear = nn.ModuleList(LinearBlock(cfg["L"]))
#         self.dropout = nn.Dropout(p=0.2)
        
#     def forward(self, data):
#         x = data.view(self.size)
#         output = []
#         for module in self.feature:
#             x = module(x.detach())
#             output.append(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         for module in self.linear:
#             x = module(x.detach())
#             output.append(x)
#         return x, output