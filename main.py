import numpy as np
import pandas as pd

import argparse


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, transforms

# from dataloader import * # dataloader 셋팅 필요!
from utils import *

def define():

    p = argparse.ArgumentParser()

    p.add_argument('--split_num', type = int, default = 30000)
    p.add_argument('--batch_size', type = int, default = 64) # config.bs

    # p.add_argument('-M', type=str)
    p.add_argument('--model', type=str, default = "CNN")

    p.add_argument('--n_epochs', type = int, default = 200)
    p.add_argument('--print_iter', type=int, default = 10)
    p.add_argument('--early_stop', type = int, default = 20)

    config = p.parse_args()

    return config

def main(config):

    train_data = None #data load 
    
    # DataLoader 
    # train_loader, valid_loader = prepare_loaders(df = train_data.data, label = train_data.targets, 
    #                                              index = config.split_num, 
    #                                              batch_size = config.batch_size)
    
    # GPU Knock
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

    # Model
    if config.model == "CNN":
        model = CNN().to(device)
        print(model)

    # Loss Function and Optimizer
    loss_fn = nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # run train
    model, result = run_train(model = model, 
                                train_loader = train_loader, 
                                valid_loader = valid_loader, 
                                loss_fn = loss_fn, 
                                optimizer = optimizer, 
                                device = device,
                                n_epochs = config.n_epochs, 
                                print_iter = config.print_iter, 
                                early_stop = config.early_stop)
    
    # Train Valid Loss History Visualization
    # loss_plot(result)

    # Train Valid ACC History Visualization
    # acc_plot(result)
    
    #  !python train.py --index 30000 --bs 64 --model CNN --n_epochs 200 --pi 20 --es 20

if __name__ == '__main__':
    config = define()
    main(config)