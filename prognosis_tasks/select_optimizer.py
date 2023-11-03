# import timm
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("..")

def select_optimizer(opt, model, paralist='none'):
    if paralist == 'none':
        paralist = model.parameters()
    if opt.opt_name == 'Adam':
        optimizer = optim.Adam(params=paralist, lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    elif opt.opt_name == 'SGD':
        optimizer = optim.SGD(params=paralist, lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    return optimizer
