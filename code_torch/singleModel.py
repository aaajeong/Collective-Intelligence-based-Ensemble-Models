import torch
import torchvision
import os
import prepareData
import Models
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')
log_dir = "../log_dir/single"

def trainSingleModel(dataset, epochs, batch_size, n_class):
    
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
    
    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                                download=True, transform=transform_1ch)
        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform_1ch)
    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform_3ch)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=transform_3ch)
    else:
        print('존재하지 않는 데이터')
    
    # DataLoaer
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers = 6)  
    
    # Train Model!
    # model = Models.Net(n_class).to(device)
    model = Models.CNN(n_class).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    writer = SummaryWriter(log_dir)
    
    loss_ = [] # 그래프를 그리기 위한 loss 저장용 리스트 
    n = len(train_loader) # 배치 개수

    print('Start Training')
    for epoch in range(epochs):  # 10번 학습을 진행한다.

        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0)):

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
        print('[%d] loss: %.3f' %(epoch + 1, running_loss / len(train_loader)))

    print('Finished Training')
    
    writer.flush()
    writer.close()
    
    return model