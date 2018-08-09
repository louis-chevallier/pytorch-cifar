from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from models import *
from utils import progress_bar
from main import *

def gather_flat_params(model):
    views = []
    for name, p in model.named_parameters() :
        view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def distribute_flat_params(model, params):
    offset = 0
    for name, p in model.named_parameters() :
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.data = params[offset:offset + numel].view_as(p.data)
        offset += numel

def mcmc(epoch, modeln = 2):

    net, optimizer, criterion = buildnet(modeln)
    net1, _, criterion1 = buildnet(modeln)    
    
    net.eval()
    net1.eval()    
    test_loss = 0
    correct = 0
    total = 0
    FF = 400.
    current = gather_flat_params(net)
    shp = current.size()
    dist = torch.distributions.studentT.StudentT(3)
    prop = current + dist.rsample(shp).to(device)/100.
    print(np.std(current.cpu().numpy()))
    print(np.std(prop.cpu().numpy()))
    print(np.std(dist.rsample(shp).cpu().numpy()/100.))

    nepochs = 1000
    temp = np.linspace(400., 4000., nepochs)
    
    for epoch in range(start_epoch, start_epoch+1000):
        train_loss = 0    
        correct = 0
        total = 0
        accepted = 0
        nb = 0
        FF = temp[epoch]
        FF = 400.
        print('FF ', FF)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                distribute_flat_params(net, current)            
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                prop = current + dist.rsample(shp).to(device)/FF
                
                distribute_flat_params(net1, prop)    
                outputs = net1(inputs)
                loss1 = criterion1(outputs, targets)
                nb += 1
                if loss1 < loss :
                    #print('loss ', loss.item(), ' ', loss1.item())
                    current = prop
                    distribute_flat_params(net, prop)
                    accepted += 1
                    loss = loss1
                train_loss += loss.item()
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | acc %d/%d'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, accepted, nb))
                    
        test(epoch, net, criterion)

if __name__ == "__main__":
    if False :
        net, optimizer, criterion = buildnet(0)
        prms = gather_flat_params(net)
        print(prms.cpu().numpy().shape)


        for epoch in range(start_epoch, start_epoch+200):
            train(epoch, net, optimizer, criterion)
            test(epoch, net, criterion)

    net, optimizer, criterion = buildnet(2)
    mcmc(1, 0)
    
    model = net
    plt.hist(prms, bins='auto'); plt.show()
    test(0)        

    distribute_flat_params(model, prms)
    test(0)        

    distribute_flat_params(model, prms*0)
    test(0)        

    distribute_flat_params(model, prms)    



    mcmc(1, model)

        
