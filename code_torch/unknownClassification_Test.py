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
from torch.utils.data.dataset import random_split
from sklearn.metrics import classification_report

# =========================================================== #
# 모델 설명
# 3채널(컬러)
# =========================================================== #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')
model_PATH = '../model/unknown_class/mnist_unknownclsfi_random_5000.h5'
# model_PATH = '../model/single/mnist_epoch300.h5'

def TestModel(dataset, batch_size, n_class):

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
    
    # Test dataset: MNIST(5,000개), EMNIST, CIFAR-10 데이터 중 5,000개를 무작위로 추출 -> 총 10,000개
    # Load data for Test
    mnistset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform_1ch)
    mnistset, _ = random_split(mnistset, [5000, len(mnistset)-5000])
    cifar10set = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=transform_3ch)
        
    emnistset = torchvision.datasets.EMNIST(root='../data', split = 'letters', train=False,
                                                download=True, transform=transform_1ch)
    
    # Preprocessing unknown data and concatenate them
    if dataset == 'mnist':
        mainset = mnistset
        unknown_list = prepareData.labelTounknown([cifar10set, emnistset], n_class)
        unknownset = torch.utils.data.ConcatDataset(unknown_list)
    elif dataset == 'cifar10':
        mainset = cifar10set
        unknown_list = prepareData.labelTounknown([mnistset, emnistset], n_class)
        unknownset = torch.utils.data.ConcatDataset(unknown_list)
    else:
        print('존재하지 않는 데이터')
        
    # Get final trainset
    final_unknown, _ = random_split(unknownset, [5000, len(unknownset)-5000])
    final_testset = torch.utils.data.ConcatDataset([mainset, final_unknown])
     
    # DataLoader
    final_testloader = torch.utils.data.DataLoader(final_testset, batch_size, shuffle=False)
    print(final_testloader)

#     # Test Model!
#     model = torch.load(model_PATH).to(device)

#     correct = 0
#     total = 0
#     model.eval()
#     pred = []
#     true = []
#     with torch.no_grad():
#         for data in tqdm(final_testloader):
#             inputs, labels = data[0].to(device), data[1].to(device)
#             # 신경망에 이미지를 통과시켜 출력을 계산
#             outputs = model(inputs)
#             # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             pred.append(predicted)
#             true.append(labels)

#     print(f'Accuracy of the network on the test images: {100 * correct // total} %')
#     print(classification_report(true, pred))

#     #  # 이미지 보여주기
#     # utils.imshow(torchvision.utils.make_grid(images))
    
#     # # # 정답(label) 출력
#     # for i in range(4):
#     #     print(labels[i], end = ' ')
#     # # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(5))) # unknown 이미지는 레이블이 class 외에 속하기 때문에 인덱스 에러 발생할 수 있으므로 주의.
#     # print()
#     # # # 예측 값 출력
#     # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(5)))

#     # Save Results
#     # if not os.path.isdir('../results/unknown_class'):
#     #     os.mkdir('../results/unknown_class')
#     # torch.save(model, '../model/unknown_class/'+dataset+'_dataselection_'+selection+'.h5')

# # dataset, batch_size, n_class
TestModel('mnist', 128, 10)
#%%