'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batchsize', default=2000, type=int)
args = parser.parse_known_args()[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class SNetAvg(nn.Module):
    def __init__(self, np=10):
        super(SNetAvg, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=2, dilation=2)
        self.conv2 = nn.Conv2d(10, 60, kernel_size=3, stride=1, dilation=2)
        self.conv3 = nn.Conv2d(60, np, kernel_size=3, stride=1, dilation=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(np, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        relu = lambda x : F.relu(x)
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        #print(x.size())
        x = relu(self.conv3(x))
        b,d,c,r = x.size()
        x = x.view(b,d,-1)
        x = torch.mean(x, 2)
        """
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = relu(self.fc2(x))
        x = relu(self.fc(x))
        """
        #return x
        return F.softmax(x, dim=1)

class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(576, 10)
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        relu = lambda x : F.relu(x)
        do = lambda x : self.conv2_drop(x)
        don = lambda x : x
        x = don(relu(self.conv1(x)))
        x = don(relu(self.conv2(x)))
        x = do(relu(self.conv3(x)))
        b,d,c,r = x.size()
        x = x.view(b,-1)
        #print(x.size())
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = relu(self.fc(x))
        return F.softmax(x, dim=1)

# Model
print('==> Building model..')


# un gros choix de reseaux

nets = [
    SNetAvg,    
    SNet,    
    lambda  : VGG('VGG19'),
    ResNet18,
    PreActResNet18,
    GoogLeNet,
    DenseNet121,
    ResNeXt29_2x64d,
    MobileNet,
    MobileNetV2,
    DPN92,
    ShuffleNetG2,
    SENet18,
    torchvision.models.AlexNet, 	#	43.45	20.91
    torchvision.models.vgg11, 	#	30.98	11.37
    torchvision.models.vgg11_bn, 	#	30.98	11.37
    torchvision.models.vgg13, 	#	30.07	10.75
    torchvision.models.vgg13_bn, 	#	30.07	10.75
    torchvision.models.vgg16, 	#	30.07	10.75
    torchvision.models.vgg16_bn, 	#	30.07	10.75
    torchvision.models.vgg19, 	#	30.07	10.75
    torchvision.models.vgg19_bn, 	#	30.07	10.75
    torchvision.models.resnet18, 	#	30.24	10.92
    torchvision.models.resnet34, 	#	30.24	10.92
    torchvision.models.resnet50, 	#	30.24	10.92
    torchvision.models.resnet101, 	#	30.24	10.92
    torchvision.models.resnet152, 	#	30.24	10.92

    torchvision.models.squeezenet1_0, 	#	41.90	19.58
    torchvision.models.squeezenet1_1, 	#	41.90	19.58
    torchvision.models.densenet121, 	#	25.35	7.83
    torchvision.models.densenet169, 	#	24.00	7.00
    torchvision.models.densenet201, 	#	22.80	6.43
    torchvision.models.inception_v3, 		# v3
]


def buildnet(n) :
    """
    instanciation du reseau
    """
    net = nets[n]().to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr)
    return net, optimizer, criterion

net, optimizer, criterion = buildnet(2)

# Training
def train(epoch, _net, _opt, _crit, call=None):
    print('\nEpoch: %d' % epoch)
    _net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        _opt.zero_grad()
        outputs = _net(inputs)
        loss = _crit(outputs, targets)

        if not call is None :
            call()
        
        
        loss.backward()
        _opt.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, _net, _crit):
    global best_acc
    _net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = _net(inputs)
            loss = _crit(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, net, optimizer, criterion)
        test(epoch, net, criterion)
