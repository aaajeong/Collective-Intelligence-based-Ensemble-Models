#%%
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import prepareData
import Models
import utils
import os
from tqdm import tqdm

# =========================================================== #
# 모델 설명
# 3채널(컬러)
# =========================================================== #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')
PATH = '../model/test/mnist_dataselection_random.h5'

def TestModel(dataset, batch_size):

    
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
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,  # train=False로 설정
                                            download=True, transform=transform_3ch)
        unknown = 'mnist'
        
    elif dataset == 'mnist':
        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform_1ch)
        unknown = 'cifar10'

    else:
        print('존재하지 않는 데이터')
        return 0
    
    
    # Load data for train: Main Unknown
    if unknown == 'mnist':
        unk_testset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform_1ch)
    else:   # unknown == 'cifar10'
        unk_testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=transform_3ch)
    
    # Load data for train: Sub Unknown
    # Sub unknown: EMNIST, CIFAR100, Imagenet
    
    # 1. EMNIST
    emn_testset = torchvision.datasets.EMNIST(root='../data', split = 'letters', train=False,
                                            download=True, transform=transform_1ch)
    
    # 2. CIFAR100
    cf100_testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                            download=True, transform=transform_3ch)
    
    # 3. Imagenet
    imgn_testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                            download=True, transform=transform_3ch)
    
    
    # Preprocessing data: concatenation
    final_testset = torch.utils.data.ConcatDataset([testset, unk_testset, emn_testset, cf100_testset, imgn_testset])
    
    # DataLoader
    final_testloader = torch.utils.data.DataLoader(final_testset, batch_size, shuffle=True)
    
    # # 학습용 이미지를 무작위로 가져오기
    # dataiter = iter(final_testloader)
    # images, labels = next(dataiter)
    # images = images.to(device)
    # labels = labels.to(device)

    # Test Model!
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown')
    
    model = torch.load(PATH).to(device)

    correct = 0
    total = 0
    
    # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없다.
    with torch.no_grad():
        for data in tqdm(final_testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            # 신경망에 이미지를 통과시켜 출력을 계산
            outputs = model(inputs)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')


    #  # 이미지 보여주기
    # utils.imshow(torchvision.utils.make_grid(images))
    
    # # # 정답(label) 출력
    # for i in range(4):
    #     print(labels[i], end = ' ')
    # # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(5))) # unknown 이미지는 레이블이 class 외에 속하기 때문에 인덱스 에러 발생할 수 있으므로 주의.
    # print()
    # # # 예측 값 출력
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(5)))

    # Save Results
    # if not os.path.isdir('../results/unknown_class'):
    #     os.mkdir('../results/unknown_class')
    # torch.save(model, '../model/unknown_class/'+dataset+'_dataselection_'+selection+'.h5')
    
TestModel('mnist', 128)
#%%