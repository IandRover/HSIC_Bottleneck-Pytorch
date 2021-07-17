#coding=utf8
"""
# Author : Jianbai(Gus) Ye
# created at Feb 2 2019
# pytorch implementation of HSIC bottleneck method
# reference : https://github.com/forin-xyz/Keras-HSIC-Bottleneck
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import torch
from torch import nn, optim
import numpy as np
import sys
import matplotlib.pyplot as plt
from torchsummary import summary
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch.nn.functional as F
import time
from utils import *
import argparse



class HSICBottleneck:
    def __init__(self, args):
        self.model      = MLP(args)
        self.model.to(device)
        self.batch_size = args.batchsize
        self.lambda_0   = args.lambda_
        self.sigma      = args.sigma_
        self.extractor  = 'hsic'
        self.last_linear = "output_layer"
        self.HSIC = compute_HSIC(args.HSIC)
        self.kernel = compute_kernel()
        self.kernel_forward = args.kernel_forward
        self.kernel_backward = args.kernel_backward
        self.forward = args.forward
        
        self.opt = optim.AdamW(self.model.parameters(), lr=0.001)
        self.iter_loss1, self.iter_loss2, self.iter_loss3 = [], [], []
        self.track_loss1, self.track_loss2, self.track_loss3 = [], [], []
        
        self.loss = args.loss
        if self.loss == "mse": self.output_criterion = nn.MSELoss()
        elif self.loss == "CE": self.output_criterion = nn.CrossEntropyLoss()
        
    def step(self, input_data, labels):
        
        self.model.train()
        labels_float = F.one_hot(labels, num_classes=10).float()
        
        if self.forward == "x": Kx  = self.kernel(input_data, self.sigma, self.kernel_forward)
        Ky = self.kernel(labels_float, self.sigma, self.kernel_backward) 
        y_pred, hidden_zs = self.model(input_data)
        
        total_loss1, total_loss2, total_loss3 = 0., 0., 0.
        kernel_list = list()
        for num, feature in enumerate(hidden_zs): kernel_list.append(self.kernel(feature, self.sigma, self.kernel_forward))
        
        for num, feature in enumerate(kernel_list):
            if num == (len(hidden_zs)-1): 
                loss1 = self.HSIC(feature, kernel_list[num-1], self.batch_size, device)
                if self.loss == "mse": total_loss3 += self.output_criterion(hidden_zs[-1], labels_float)
                if self.loss == "CE": total_loss3 += self.output_criterion(hidden_zs[-1], labels)
            elif num == 0:
                if self.forward == "x": total_loss1 += self.HSIC(feature, Kx, self.batch_size, device)
                total_loss2 += - self.lambda_0*self.HSIC(feature, Ky, self.batch_size, device)
            else:
                if self.forward == "f": total_loss1 += self.HSIC(feature, kernel_list[num-1], self.batch_size, device)
                elif self.forward == "x": total_loss1 += self.HSIC(feature, Kx, self.batch_size, device)
                total_loss2 += - self.lambda_0*self.HSIC(feature, Ky, self.batch_size, device)
        
        if self.forward == "f" or self.forward == "x": 
            total_loss = total_loss1 + total_loss2 + total_loss3
            self.iter_loss1.append(total_loss1.item())
        if self.forward == "n": 
            total_loss = total_loss2 + total_loss3
            self.iter_loss1.append(-1)
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
                
        self.iter_loss2.append(total_loss2.item())
        self.iter_loss3.append(total_loss3.item())
        
    def update_loss(self):
        self.track_loss1.append(np.mean(self.iter_loss1))
        self.track_loss2.append(np.mean(self.iter_loss2))
        self.track_loss3.append(np.mean(self.iter_loss3))
        self.iter_loss1, self.iter_loss2, self.iter_loss3 = [], [], []
    
    def tune_output(self, input_data, labels):
        self.model.train()
        if self.loss == "mse":
            one_hot_labels = F.one_hot(labels, num_classes=10)
            labels = F.one_hot(labels, num_classes=10).float()
        
        y_pred, hidden_zs = self.model(input_data)
        total_loss = self.output_criterion(hidden_zs[-1], labels)
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
    
def show_result():
    hsic.model.eval()
    with torch.no_grad():
        counts, correct, counts2, correct2 = 0, 0, 0, 0        
        for batch_idx, (data, target) in enumerate(train_loader): 
            output = hsic.model.forward(data.view(batch_size, -1).to(device))[0].cpu()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred[:,0] == target).float().sum()
            counts += len(pred)
        for batch_idx, (data, target) in enumerate(test_loader): 
            output = hsic.model.forward(data.view(batch_size, -1).to(device))[0].cpu()
            pred = output.argmax(dim=1, keepdim=True)
            correct2 += (pred[:,0] == target).float().sum()
            counts2 += len(pred)
            
        train_acc =  np.round(correct/counts, 4).numpy()
        test_acc =  np.round(correct2/counts2, 4).numpy()
        print("Training  ACC: {:.4f} \t Testing ACC: {:.4f}".format(train_acc, test_acc))
        logs.append([epoch, train_acc, test_acc])
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default="CE")
    parser.add_argument('--HSIC', type=str, default="nHSIC")
    parser.add_argument('--kernel_forward', type=str, default="student", choices=["rbf", "student"])
    parser.add_argument('--kernel_backward', type=str, default="rbf", choices=["rbf", "student"])
    parser.add_argument('--sigma_', type=int, default=1)
    parser.add_argument('--lambda_', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bn_affine', type=int, default=0)
    parser.add_argument('--forward', type=str, default="f", choices=["x", "f", "n"])
    args, _ = parser.parse_known_args()    
    
    print(get_filename(args))
    
    torch.manual_seed(1)
    device = "cuda:{}".format(args.device)
    batch_size = args.batchsize
    train_loader, test_loader = load_data(batch_size=args.batchsize)
    
    logs = list()
    hsic = HSICBottleneck(args)
    start = time.time()
    for epoch in range(50):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(batch_size, -1)
            hsic.step(data.view(batch_size, -1).to(device), target.to(device))
            hsic.tune_output(data.view(batch_size, -1).to(device), target.to(device))
        if epoch in range(10) or epoch in range(0, 100, 10):
            print("EPOCH %d" % epoch)
            show_result()
            print("{:.2f}".format(time.time()-start))
            start = time.time()
    filename = get_filename(args)
    
    txt_path = os.path.join("./results", filename+".csv")
    df = pd.DataFrame(logs)
    df.to_csv(txt_path,index=False)