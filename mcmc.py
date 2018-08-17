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

import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
from math import pow, log

_once=0
def once() :
    global _once
    if _once == 0 :
        _once += 1
        return True
    else:
        return False

def gather_flat_params(model):
    """
    recupere tous les coef d'un modele sous la forme d'un seul vecteur
    """
    views = []
    for name, p in model.named_parameters() :
        view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def distribute_flat_params(model, params):
    """
    affecte les coef du modele a partir d'un vecteur
    """

    offset = 0
    for name, p in model.named_parameters() :
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.data = params[offset:offset + numel].view_as(p.data)
        offset += numel

def mcmc(epoch, net, criterion, netinit=None):

    # pas la peine de calculer la backprop, on met en mode eval (juste la forward est necessaire)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    FF = 400.

    # a la creation du modele les layers sont initialises avec des bonnes valeurs (en fct de la taille des filtres etc.)
    # on recupere ca pour initiliser le pt de depart
    current = gather_flat_params(net if netinit is None else netinit)
    shp = current.size()

    # pour la generation des deplacement aleatoires
    dist = torch.distributions.studentT.StudentT(2)

    prop = current + dist.rsample(shp).to(device)/100.
    print(np.std(current.cpu().numpy()))
    print(np.std(prop.cpu().numpy()))
    print(np.std(dist.rsample(shp).cpu().numpy()/100.))

    nepochs = 1000
    temp = np.linspace(1000., 10000., nepochs)
    
    for epoch in range(start_epoch, start_epoch+nepochs):
        train_loss = 0    
        correct = 0
        total = 0
        accepted = 0
        nb = 0
        FF = temp[epoch]
        FF = 10000.
        print('FF %d, epoch %d ' % (FF, epoch))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                if once() :
                    with SummaryWriter(comment='Net1') as w:
                        w.add_graph(net, (inputs, ))
                
                distribute_flat_params(net, current)

                # forward avec le current
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                for iprop in range(16) :
                    # on divise l'alea pour diminuer la variance, pour ne pas se deplacer trop vite ds l'espace
                    prop = current + dist.rsample(shp).to(device)/FF

                    distribute_flat_params(net, prop)
                    # forward avec la proposition
                    outputs = net(inputs)
                    loss1 = criterion(outputs, targets)

                    nb += 1
                    r1 = min(loss1/loss, 1.)
                    r = pow(r1,2000)                    
                    alea = np.random.uniform()
                    accept = r < alea  and alea < 1
                    """
                    print('loss %f loss1 %f r1 %f r %f log %f alea %f acc %d' % (loss.item(),loss1.item(),
                                                                                 r1, r,
                                                                                 log(r),
                                                                                 alea, accept))
                    """
                    if accept : 
                        current = prop
                        distribute_flat_params(net, prop)
                        accepted += 1
                        loss = loss1
                        break
                train_loss += loss.item()
                writer.add_scalar('loss', train_loss) 
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | acc %d/%d'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, accepted, nb))
        # eval  sur le test set
        test(epoch, net, criterion)

if __name__ == "__main__":
    writer = SummaryWriter()
    net, optimizer, criterion = buildnet(1)
    
    if False :
        # test du reseau avec la sgd
        prms = gather_flat_params(net)
        print(prms.cpu().numpy().shape)
        print(str(net))
        ne = 30
        def call() :
            p = gather_flat_params(net)
            #print( p.size())
        for epoch in range(start_epoch, start_epoch+ne):
            train(epoch, net, optimizer, criterion, call)
            test(epoch, net, criterion)
            writer.add_scalar('acc', best_acc)
        state = {
            'net': net.state_dict(),
        }
        torch.save(state, 'ckpt.t7')
            
    else :
        checkpoint = torch.load('ckpt.t7')
        net.load_state_dict(checkpoint['net'])
            

    mcmc(1, net, criterion, net)    
    writer.close()

        
