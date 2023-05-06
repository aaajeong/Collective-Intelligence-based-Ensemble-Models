import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import json
import os

# For 3 channel
transform_3ch = transforms.Compose(
    [transforms.ToTensor(), # 텐서로 바꿔주고([-1,1])
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 3개의 채널에 대한 평균, 표준편차를 넣어준다.(정규화)(값은 최적화 값은 아님)

trainset = torchvision.datasets.ImageNet('../data/imagenet', split='train', download=None, transform=transform_3ch)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.ImageNet('../data/imagenet', split='val', download=None, transform=transform_3ch)
print(len(testset))
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)