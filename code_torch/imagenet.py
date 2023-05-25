import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import prepareData
import json
import os
from tqdm import tqdm
from sklearn.metrics import classification_report
import customDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

# For 3 channel
transform_3ch = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

target_transform = transforms.Lambda(lambda y: 10 if y >= 0 else 10)
# trainset = torchvision.datasets.ImageNet('../data/imagenet', split='train', download=None, transform=transform_3ch)
# print(len(trainset))
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# testset = torchvision.datasets.ImageNet('../data/imagenet', split='val', download=None, transform=transform_3ch, target_transform = target_transform)


# cifar10set = torchvision.datasets.CIFAR10(root='../data', train=False,
#                                             download=True, transform=transform_3ch)
# cifarloader = torch.utils.data.DataLoader(cifar10set, 128, shuffle=False)
# unknown_cifar = prepareData.labelTounknown([cifar10set], 10)
# unknownset = torch.utils.data.ConcatDataset(unknown_cifar)
# cifarloader = torch.utils.data.DataLoader(unknownset, 128, shuffle=False)

unknown_img = prepareData.labelTounknown([testset], 10)
unknownset = torch.utils.data.ConcatDataset(unknown_img)
imgloader = torch.utils.data.DataLoader(unknownset, 128, shuffle=False)

# print(testset.targets)
# print(testset.image)
# print(testset[1])
# new_testset = customDataset.ImgnetCustomDataset(testset.images, testset.targets, 10)
# newloader = torch.utils.data.DataLoader(new_testset, 128, shuffle=False)

# print(new_testset.targets)

dataiter = iter(imgloader)
inputs, labels = next(dataiter)
print(inputs)
print(labels)
    
# # Test Model!
# model_PATH = "../model/unknown_class/cifar10_unknownclsfi_random_100.h5"
# model = torch.load(model_PATH).to(device)

# correct = 0
# total = 0
# model.eval()
# pred = []
# true = []
# cnt = 0
# with torch.no_grad():
#     for data in tqdm(final_testloader):
#         cnt += 1
#         inputs, labels = data[0].to(device), data[1].to(device)
#         # 신경망에 이미지를 통과시켜 출력을 계산
#         outputs = model(inputs)
#         # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
#         pred = np.concatenate((pred, predicted.cpu()), axis = 0)
#         true = np.concatenate((true, labels.cpu()), axis = 0)

# # Save Results
# f = open('test.txt', 'w')
# f.write(f'Accuracy of the network on the test images: {100*correct // total:.3f} %')
# f.write('\n\n')
# f.write(classification_report(true, pred))
# f.close()