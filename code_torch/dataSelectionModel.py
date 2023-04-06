import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# =========================================================== #
# 모델 설명
# 3채널(컬러)
# =========================================================== #

# def trainModel(dataset, n_class, selection, epochs, batch_size):
def trainModel(dataset, batch_size):
    
    # 전처리 진행
    
    # For 1 channel
    transform_1ch = transforms.Compose(
        [transforms.Resize((32, 32)), # 사이즈 통일
         transforms.ToTensor(), # 텐서로 바꿔주고([-1,1])
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),    # 1개의 채널을 3개의 채널로 바꿔준다.
         transforms.Normalize((0.5), (0.5))])   # 3개의 채널에 대한 평균, 표준편차를 넣어준다.(정규화)(값은 최적화 값은 아님)
    
    # For 3 channel
    transform_3ch = transforms.Compose(
        [transforms.ToTensor(), # 텐서로 바꿔주고([-1,1])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 3개의 채널에 대한 평균, 표준편차를 넣어준다.(정규화)(값은 최적화 값은 아님)
    
        
    # Load data for train: Known
    if dataset == 'cifar10':
        
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform_3ch) # transform을 이용해서 전처리 한다.
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,  # train=False로 설정
                                            download=True, transform=transform_3ch)
        unknown = 'mnist'
        
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                                download=True, transform=transform_1ch)
        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform_1ch)
        unknown = 'cifar10'

    else:
        print('존재하지 않는 데이터')
        return 0
    
    
    # Load data for train: Main Unknown
    if unknown == 'mnist':
        unk_trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                                download=True, transform=transform_1ch)
        unk_testset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform_1ch)
    else:   # unknown == 'cifar10'
        unk_trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform_3ch)
        unk_testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=transform_3ch)
    
    # Load data for train: Sub Unknown
    # Sub unknown: EMNIST, CIFAR100, Imagenet
    
    # 1. EMNIST
    emn_trainset = torchvision.datasets.EMNIST(root='../data', split = 'letters', train=True,
                                                download=True, transform=transform_1ch)
    emn_testset = torchvision.datasets.EMNIST(root='../data', split = 'letters', train=False,
                                            download=True, transform=transform_1ch)
    
    # 2. CIFAR100
    cf100_trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                                download=True, transform=transform_3ch)
    cf100_testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                            download=True, transform=transform_3ch)
    
    # 3. Imagenet
    imgn_trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                                download=True, transform=transform_3ch)
    imgn_testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                            download=True, transform=transform_3ch)
    
    # DataLoader
    # 1. Known dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True) 
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)
    
    # # 2. Unknown Main dataset
    # unk_trainloader = torch.utils.data.DataLoader(unk_trainset, batch_size, shuffle=True) 
    # unk_testloader = torch.utils.data.DataLoader(unk_testset, batch_size, shuffle=False)
    
    # # 3. Unknown Sub dataset
    # emn_trainloader = torch.utils.data.DataLoader(emn_trainset, batch_size, shuffle=True) 
    # emn_testloader = torch.utils.data.DataLoader(emn_testset, batch_size, shuffle=False)
    
    # cf100_trainloader = torch.utils.data.DataLoader(cf100_trainset, batch_size, shuffle=True) 
    # cf100_testloader = torch.utils.data.DataLoader(cf100_testset, batch_size, shuffle=False)
    
    # imgn_trainloader = torch.utils.data.DataLoader(imgn_trainset, batch_size, shuffle=True) 
    # imgn_testloader = torch.utils.data.DataLoader(imgn_testset, batch_size, shuffle=False)
    
    
    # Concatenate unknown datas
    # Final Unknwon data = unknown + emnist + cifar100 + imagenet
    """
    <MNIST>
    images:  torch.Size([128, 1, 28, 28]) -> torch.Size([128, 3, 32, 32])
 
    <CIFAR10>
    images: torch.Size([128, 3, 32, 32])
    
    <CIFAR100>
    images: torch.Size([128, 3, 32, 32])
    
    <EMNIST>
    images: torch.Size([128, 1, 28, 28]) -> torch.Size([128, 3, 32, 32])
    
    <Imagenet>
    images: torch.Size([128, 3, 32, 32])
    
    """
    print('mnist train: ', len(trainset))
    print('cifar10 train: ', len(unk_trainset))
    print('emnist train: ', len(emn_trainset))
    print('cifar100 train: ', len(cf100_trainset))
    print('imagenet train: ', len(imgn_testset))
    
    print('mnist test: ', len(testset))
    print('cifar10 test: ', len(unk_testset))
    print('emnist test: ', len(emn_testset))
    print('cifar100 test: ', len(cf100_testset))
    print('imagenet test: ', len(imgn_testset))
    
    final_unknwon_train = torch.utils.data.ConcatDataset([unk_trainset, emn_trainset, 
                                                    cf100_trainset, imgn_trainset])
    final_unknwon_test = torch.utils.data.ConcatDataset([unk_testset, emn_testset, 
                                                    cf100_testset, imgn_testset])
    
    final_unk_trainloader = torch.utils.data.DataLoader(final_unknwon_train, batch_size, shuffle=True)
    final_unk_testloader = torch.utils.data.DataLoader(final_unknwon_test, batch_size, shuffle=False)
    
    
    """
    final_unknwon_train length:  274800
    final_unknwon_test length:  50800
    images:  torch.Size([128, 3, 32, 32])
    lables:  torch.Size([128])
    """
    num = 5000
    

trainModel('mnist', 128)