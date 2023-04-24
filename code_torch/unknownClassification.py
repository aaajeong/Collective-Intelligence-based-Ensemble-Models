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
import os
from tqdm import tqdm
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
 

# =========================================================== #
# 모델 설명
# 3채널(컬러)
# =========================================================== #

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')
log_dir = "../log_dir"

def trainModel(dataset, n_class, selection, epochs, batch_size):
# def trainModel(dataset, batch_size):
    
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
        unknown = 'imagenet'
        
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                                download=True, transform=transform_1ch)
        trainset, _ = random_split(trainset, [50000, len(trainset)-50000])
        
        unknown = 'cifar10'

    else:
        print('존재하지 않는 데이터')
        return 0
    
    
    # Load data for train: Main Unknown
    if unknown == 'cifar10':
        unk_trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform_3ch)
    else:   # unknown == 'imagenet'
        unk_trainset = torchvision.datasets.ImageNet(root='../data', train=True,
                                                download=True, transform=transform_3ch)
    
    
    # Preprocessing unknown data and concatenate them
    unknown_list = prepareData.labelTounknown([unk_trainset], n_class)
    final_unknown = torch.utils.data.ConcatDataset(unknown_list)
    
    
    # You can adjust the number of unknown data
    num = 5000
    
    # Get final trainset
    final_trainset = prepareData.unknownClassData(dataset, trainset, n_class, final_unknown, num, selection)
    
    
    # DataLoader
    final_trainloader = torch.utils.data.DataLoader(final_trainset, batch_size, shuffle=True, num_workers = 6)     
    
    """
    <MNIST>
    images:  torch.Size([128, 1, 28, 28]) -> torch.Size([128, 3, 32, 32])
 
    <CIFAR10>
    images: torch.Size([128, 3, 32, 32])
    
    <Imagenet>
    images: torch.Size([128, 3, 32, 32])
    
    """

    # Train Model!
    model = Models.Net(n_class).to(device)
    # model = Models.CNN(n_class).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    writer = SummaryWriter(log_dir)
    
    loss_ = [] # 그래프를 그리기 위한 loss 저장용 리스트 
    n = len(final_trainloader) # 배치 개수

    print('Start Training')
    for epoch in range(epochs):  # 10번 학습을 진행한다.

        running_loss = 0.0
        for i, data in tqdm(enumerate(final_trainloader, 0)):

            # gpu용 데이터와 모델이 있어야 함. (to(device)를 해줘야 한다.)
            inputs, labels = data[0].to(device), data[1].to(device) # 배치 데이터 
            
            # 변화도(Gradient) 매개변수를 0으로.
            optimizer.zero_grad()

            outputs = model(inputs) # 예측값 산출 
            loss = criterion(outputs, labels) # 손실함수 계산
            
            writer.add_scalar("Loss/train", loss, epoch)
            
            loss.backward() # 손실함수 기준으로 역전파 선언
            optimizer.step() # 가중치 최적화

            # print statistics
            running_loss += loss.item()

        loss_.append(running_loss / n)    
        print('[%d] loss: %.3f' %(epoch + 1, running_loss / len(final_trainloader)))

    print('Finished Training')
    
    writer.flush()
    writer.close()
   
    # Save Model
    if not os.path.isdir('../model/unknown_class'):
        os.mkdir('../model/unknown_class')
    torch.save(model, '../model/unknown_class/'+dataset+'_unknownclsfi_'+selection+'.h5')

trainModel('mnist', 10, 'random', 300, 128)

# tensorboard --logdir ./logs