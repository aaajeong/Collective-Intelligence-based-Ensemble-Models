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
import Models
import os
from torch.utils.data import Subset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

transform_1ch = transforms.Compose(
        [transforms.Resize((32, 32)), # 사이즈 통일
         transforms.ToTensor(), # 텐서로 바꿔주고([-1,1])
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),    # 1개의 채널을 3개의 채널로 바꿔준다.
         transforms.Normalize((0.5), (0.5))])   # 3개의 채널에 대한 평균, 표준편차를 넣어준다.(정규화)(값은 최적화 값은 아님)

trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                                download=True, transform=transform_1ch)

unk_trainset = torchvision.datasets.EMNIST(root='../data', split = 'letters', train=True,
                                                   download=True, transform=transform_1ch)
    
# Unown class
n_class = 10
unknown_list = prepareData.labelTounknown([unk_trainset], n_class)
final_unknown = torch.utils.data.ConcatDataset(unknown_list)


indices = [4, 3, 1]
subset = Subset(final_unknown, indices)

# print(final_unknown.targets[:5])

print(subset[0])