#coding=utf8

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
        self.kernel_x = args.kernel_x
        self.kernel_h = args.kernel_h
        self.kernel_y = args.kernel_y
        self.forward = args.forward
        
        self.opt = optim.AdamW(self.model.parameters(), lr=0.001)
        self.iter_loss1, self.iter_loss2, self.iter_loss3 = [], [], []
        self.track_loss1, self.track_loss2, self.track_loss3 = [], [], []
        
        self.loss = args.loss
        if self.loss == "mse": self.output_criterion = nn.MSELoss()
        elif self.loss == "CE": self.output_criterion = nn.CrossEntropyLoss()
        
    def step(self, input_data, labels):
        
        labels_float = F.one_hot(labels, num_classes=10).float()
        if self.forward == "x": Kx  = self.kernel(input_data, self.sigma, self.kernel_x)
        Ky = self.kernel(labels_float, self.sigma, self.kernel_y) 
        
        kernel_list = list()
        y_pred, hidden_zs = self.model(input_data)
        for num, feature in enumerate(hidden_zs): kernel_list.append(self.kernel(feature, self.sigma, self.kernel_h))
        
        total_loss1, total_loss2, total_loss3 = 0., 0., 0.
        for num, feature in enumerate(kernel_list):
            if num == (len(hidden_zs)-1): 
                if self.forward == "h": total_loss1 += self.HSIC(feature, kernel_list[num-1], self.batch_size, device)
                elif self.forward == "x": total_loss1 += self.HSIC(feature, Kx, self.batch_size, device)
                if self.loss == "mse": total_loss3 += self.output_criterion(hidden_zs[-1], labels_float)
                elif self.loss == "CE": total_loss3 += self.output_criterion(hidden_zs[-1], labels)
            elif num == 0:
                if self.forward == "x": total_loss1 += self.HSIC(feature, Kx, self.batch_size, device)
                total_loss2 += - self.lambda_0*self.HSIC(feature, Ky, self.batch_size, device)
            else:
                if self.forward == "h": total_loss1 += self.HSIC(feature, kernel_list[num-1], self.batch_size, device)
                elif self.forward == "x": total_loss1 += self.HSIC(feature, Kx, self.batch_size, device)
                total_loss2 += - self.lambda_0*self.HSIC(feature, Ky, self.batch_size, device)
        
        if self.forward == "h" or self.forward == "x": 
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

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--loss', type=str, default="CE")
    parser.add_argument('--HSIC', type=str, default="nHSIC")
    parser.add_argument('--kernel_x', type=str, default="rbf", choices=["rbf", "student"])
    parser.add_argument('--kernel_h', type=str, default="rbf", choices=["rbf", "student"])
    parser.add_argument('--kernel_y', type=str, default="rbf", choices=["rbf", "student"])
    parser.add_argument('--sigma_', type=int, default=10)
    parser.add_argument('--lambda_', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bn_affine', type=int, default=0)
    parser.add_argument('--forward', type=str, default="n", choices=["x", "h", "n"])
    args, _ = parser.parse_known_args()
    filename = get_filename(args)
    print(filename)
    
    torch.manual_seed(1)
    device = "cuda:{}".format(args.device)
    batch_size = args.batchsize
    train_loader, test_loader = load_data(batch_size=args.batchsize)
    
    logs = list()
    hsic = HSICBottleneck(args)
    start = time.time()
    for epoch in range(100):
        hsic.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(args.batchsize, -1)
            hsic.step(data.view(args.batchsize, -1).to(device), target.to(device))
            hsic.tune_output(data.view(args.batchsize, -1).to(device), target.to(device))
        if epoch in range(0, 100, 10):
            show_result(hsic, train_loader, test_loader, epoch, logs, device)
            print("{:.2f}".format(time.time()-start))
            start = time.time()
            
    txt_path = os.path.join("./results", filename+".csv")
    df = pd.DataFrame(logs)
    df.to_csv(txt_path,index=False)