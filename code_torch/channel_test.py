import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import prepareData
import customDataset

def trainModel(n_class, selection, batch_size):
    # 전처리 진행
    
    transform_1ch = transforms.Compose(
        [transforms.ToTensor(), # 텐서로 바꿔주고([-1,1])
        transforms.Normalize((0.5), (0.5))]) # 3개의 채널에 대한 평균, 표준편차를 넣어준다.(정규화)(값은 최적화 값은 아님)
    
    transform_1ch2 = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 1개의 채널을 3개의 채널로 바꿔준다.
    
    transform_3ch = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # transform_3chn = transforms.Compose(
    #     [transforms.Grayscale(3),   # 1채널 -> 3채널
    #      transforms.ToTensor(), # 텐서로 바꿔주고([-1,1])
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 3개의 채널에 대한 평균, 표준편차를 넣어준다.(정규화)(값은 최적화 값은 아님)
    
    
    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                                download=True, transform=transform_1ch2)
    print(len(trainset))    #60000
    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                        download=True, transform=transform_1ch2)

    
    trainset2 = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform_3ch)
    print(len(trainset2))   # 50000
    testset2 = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform_3ch)
    
    trainset3 = torchvision.datasets.CIFAR100(root='../data', train=True,
                                                download=True, transform=transform_3ch)
    print(len(trainset2))   # 5000
    testset3 = torchvision.datasets.CIFAR100(root='../data', train=False,
                                        download=True, transform=transform_3ch)
    
     # DataLoader
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
    # trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size, shuffle=True) 

    
    # final_unknown = torch.utils.data.ConcatDataset([trainset2, trainset3])
    final_unknown = customDataset.custom_wholeset([trainset2, trainset3], [torch.Tensor(trainset2.targets), torch.Tensor(trainset3.targets)])
    
    # print(len(final_unknown))   #100000
    # final_trainloader = torch.utils.data.DataLoader(final_unknown, batch_size, shuffle=True) 

    # images, labels = next(iter(final_trainloader))
    # print('images: ', images.size())
    # print('lables: ', labels.size())
    
    num = 5000

    final_trainset = prepareData.unknownClassData('mnist', trainset, n_class, final_unknown, num, selection)
    
    # final_trainloader = torch.utils.data.DataLoader(final_trainset, batch_size, shuffle=True) 
    
    
trainModel(11, 'random', 128)