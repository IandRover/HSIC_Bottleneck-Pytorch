import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F

### MODEL ###
class MLPBlock(nn.Module):
    def __init__(self, inplane, outplane, args):
        super(MLPBlock, self).__init__()
        self.linear = nn.Linear(inplane, outplane)
        
#         w = torch.nn.init.xavier_uniform_(torch.rand(outplane, inplane), gain=1.0)
#         self.linear_weight = nn.Parameter(w)
#         self.linear_bias = nn.Parameter(torch.rand(outplane))
#         self.linear_bias.data.fill_(0.1)

        w1 = torch.nn.init.xavier_uniform_(torch.rand(1, inplane), gain=1.0)
        self.linear_w1 = nn.Parameter(w1)
        w2 = torch.nn.init.xavier_uniform_(torch.rand(outplane, 1), gain=1.0)
        self.linear_w2 = nn.Parameter(w2)
        self.linear_bias = nn.Parameter(torch.rand(outplane))
        self.linear_bias.data.fill_(0.1)
        
        w1 = torch.nn.init.xavier_uniform_(torch.rand(1, inplane), gain=1.0)
        self.linear_w1_2 = nn.Parameter(w1)
        w2 = torch.nn.init.xavier_uniform_(torch.rand(outplane, 1), gain=1.0)
        self.linear_w2_2 = nn.Parameter(w2)
        
        self.sign = torch.diag(torch.sign(torch.rand(inplane))).to("cuda:{}".format(args.device))
        
        self.bn = nn.BatchNorm1d(outplane, affine=args.bn_affine)
        self.act = nn.ReLU()
        
    def forward(self, x):
#         Sign weight
#         x = F.linear(x, torch.mm(torch.abs(self.linear_weight), self.sign) , torch.abs(self.linear_bias))
#         two 相乘
#         print(torch.mm(self.linear_w2.view(1,-1), self.linear_w1))
        x = F.linear(x, torch.mm(self.linear_w2, self.linear_w1) + torch.mm(self.linear_w2_2, self.linear_w1_2) , self.linear_bias)
#         original
#         x = F.linear(x, self.linear_weight , self.linear_bias) # epoch 0: 93%
        x = self.act(x)
        x = self.bn(x)
        return x
        
class signMLP(nn.Module):
    def __init__(self, args):
        super(signMLP, self).__init__()
        self.bn_affine = True if args.bn_affine == 1 else False
        if args.dataset == "mnist":
            self.units = [784, 512, 512, 128]
            self.output_layer  = nn.Linear(self.units[-1], 10)        
        elif args.dataset == "cifar":
            self.units = [3072, 256, 256, 256, 256, 256]
            self.output_layer  = nn.Linear(self.units[-1], 10)

        self.module_list = nn.ModuleList( [MLPBlock(self.units[i], self.units[i+1], args) for i in range(len(self.units)-1)])
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