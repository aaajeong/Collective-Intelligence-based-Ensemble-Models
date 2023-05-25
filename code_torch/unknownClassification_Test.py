#%%
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

def TestModel(dataset, batch_size, n_class, selection, num):

    # 전처리 진행
    # For 1 channel
    transform_1ch = transforms.Compose(
        [transforms.Resize((32, 32)), # 사이즈 통일
         transforms.ToTensor(), # 텐서로 바꿔주고([-1,1])
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),    # 1개의 채널을 3개의 채널로 바꿔준다.
         transforms.Normalize((0.5), (0.5))])   # 3개의 채널에 대한 평균, 표준편차를 넣어준다.(정규화)(값은 최적화 값은 아님)
    
    # For 3 channel
    transform_3ch = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
     # For Imagenet
    target_transform = transforms.Lambda(lambda y: n_class if y >= 0 else n_class)
    
    
    # Test dataset: MNIST(5,000개), EMNIST, CIFAR-10 데이터 중 5,000개를 무작위로 추출 -> 총 10,000개
    # Load data for Test
    mnistset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform_1ch)
    cifar10set = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=transform_3ch)
        
    emnistset = torchvision.datasets.EMNIST(root='../data', split = 'letters', train=False,
                                                download=True, transform=transform_1ch)
    imgnetset = torchvision.datasets.ImageNet(root='../data/imagenet', split = 'val',
                                                     download=None, transform=transform_3ch, target_transform = target_transform)
    # Preprocessing unknown data and concatenate them
    if dataset == 'mnist':
        mnistset, _ = random_split(mnistset, [5000, len(mnistset)-5000])
        mainset = mnistset
        # unknown_list = prepareData.labelTounknown([cifar10set, emnistset], n_class) # 1번째에 실험한 버전
        unknown_list = prepareData.labelTounknown([emnistset], n_class) # 2번째에 실험한 버전
        unknownset = torch.utils.data.ConcatDataset(unknown_list)
    elif dataset == 'cifar10':
        mainset = cifar10set
        # unknown_list = prepareData.labelTounknown([mnistset, emnistset], n_class)   # 1번째에 실험한 버전
        unknown_list = prepareData.labelTounknown([imgnetset], n_class)   # 2번째에 실험한 버전
        unknownset = torch.utils.data.ConcatDataset(unknown_list)
    else:
        print('존재하지 않는 데이터')
        
    # Get final trainset
    final_unknown, _ = random_split(unknownset, [10000, len(unknownset)-10000])
    final_testset = torch.utils.data.ConcatDataset([mainset, final_unknown])
    
    # DataLoader
    final_testloader = torch.utils.data.DataLoader(final_testset, batch_size, shuffle=False)

    # Test Model!
    model_PATH = '../model/unknown_class/'+dataset+'_unknownclsfi_'+selection+'_'+str(num)+'.h5'
    model = torch.load(model_PATH).to(device)

    correct = 0
    total = 0
    model.eval()
    pred = []
    true = []
    cnt = 0
    with torch.no_grad():
        for data in tqdm(final_testloader):
            cnt += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            # 신경망에 이미지를 통과시켜 출력을 계산
            outputs = model(inputs)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pred = np.concatenate((pred, predicted.cpu()), axis = 0)
            true = np.concatenate((true, labels.cpu()), axis = 0)
    
    # Save Results
    m = 'unknown_class'
    if not os.path.exists('../results/'+m):
        os.makedirs('../results/'+m)
    f = open('../results/'+m+'/'+dataset+'_'+selection+'_'+str(num)+'.txt', 'w')
    f.write(f'Accuracy of the network on the test images: {100*correct // total:.3f} %')
    f.write('\n\n')
    f.write(classification_report(true, pred, digits = 4))
    f.close()

# MNIST (클래스 10개)
# single: TestModel('mnist', 128, 9)
# unknown: TestModel('mnist', 128, 10)

#CIFAR10 (클래스 10개)
# single: TestModel('cifar10', 128, 9)
# unknown: TestModel('cifar10', 128, 10)
# TestModel('cifar10', 128, 10, 'random', 500)

#%%