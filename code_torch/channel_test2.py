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
import customDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

def trainModel(epochs, batch_size):
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
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform_3ch)

    
    unk_trainset = torchvision.datasets.ImageNet(root='../data/imagenet', split = 'train',
                                                     download=None, transform=transform_3ch)
    unk_trainset = customDataset.custom_subset(unk_trainset, 50000)


    # # Unown class
    # unknown_list = prepareData.labelTounknown([unk_trainset], 10)
    # final_unknown = torch.utils.data.ConcatDataset(unknown_list)
    
    # unknown = prepareData.choiceUnknown('cifar10', final_unknown, 5000, 'random', 10, batch_size)
    
    # DataLoader
    print(unk_trainset)
    unknown_trainloader = torch.utils.data.DataLoader(unk_trainset, batch_size, shuffle=True)
    print(len(unknown_trainloader))
    
    # train_trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)

    dataiter = iter(unknown_trainloader)
    inputs, labels = next(dataiter)
    print(inputs)
    print(labels)
    
    # Train Model
    model = Models.CNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss_ = [] # 그래프를 그리기 위한 loss 저장용 리스트 
    n = len(unknown_trainloader) # 배치 개수

    # for epoch in range(epochs):  # 10번 학습을 진행한다.

    #     running_loss = 0.0
    #     for i, data in enumerate(unknown_trainloader, 0):

    #         # gpu용 데이터와 모델이 있어야 함. (to(device)를 해줘야 한다.)
    #         inputs, labels = data[0].to(device), data[1].to(device) # 배치 데이터 
            
    #         # 변화도(Gradient) 매개변수를 0으로.
    #         optimizer.zero_grad()

    #         outputs = model(inputs) # 예측값 산출 
    #         loss = criterion(outputs, labels) # 손실함수 계산
    #         loss.backward() # 손실함수 기준으로 역전파 선언
    #         optimizer.step() # 가중치 최적화

    #         # print statistics
    #         running_loss += loss.item()

    #     loss_.append(running_loss / n)    
    #     print('[%d] loss: %.3f' %(epoch + 1, running_loss / len(unknown_trainloader)))

    # print('Finished Training')
    
    # Save Model
    # if not os.path.isdir('../model/test'):
    #     os.mkdir('../model/test')
    # torch.save(model, '../model/test/'+dataset+'_dataselection_'+selection+'.h5')
    
trainModel(100, 128)