import math
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import RandomSampler
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getModel(main_dataset):
    model_path = '../model/single'+ main_dataset + '_epoch300.h5'
    model = torch.load(model_path)
    return model

def checkData(dataset, selection, data_list):
    """
    dataset: mnist or cifar10
    selection: unknown 선택 기준 방법
    data_list: unknown 데이터를 랜덤 비복원 추출 한 인덱스 리스트
    """
    data_num = {'mnist':0, 'emnist':0, 'cifar10':0, 'cifar100':0, 'imagenet':0}
    
    
    
# 엔트로피를 말로 설명하면, 모든 사건 정보량의 기대값
# entropy가 높을수록 불확실성이 높다.
def getEntropyBasedUncertainty(output, n_class):
    """
    output:  single model이 unknown dataset에 대해 예측한 output
    n_class: unknown 클래스 번호
    """
    uncertainty = []
    for d in output:
        tmp = 0
        # d[0]~d[n_class-1] 까지의 확률값에 대한 엔트로피를 구해 d에 대한 uncertainty 구한다.
        # log: base가 n_class
        for i in range(n_class):
            tmp = tmp + d[i]*math.log(d[i],n_class) if d[i] != 0 else tmp
        uncertainty.append(-1*tmp)
    return uncertainty


def choiceRandom(k, unknown):
    """
    k:unknown dataset 개수 설정한 값
    unknown: unknown 데이터
    n_class: unknown 클래스 번호
    """
    
    # k 만큼 분리
    # <torch.utils.data.dataset.Subset object at 0x7fba227f4610>
    chosen_data, _ = random_split(unknown, [k, len(unknown)-k])

    return chosen_data


def choiceTopk(dataset, k, unknown, n_class, batch_size):
    """
    dataset: known dataset
    k:unknown dataset 개수 설정한 값
    unknown: unknown 데이터
    n_class: unknown 클래스 번호
    """
    pred_unknown = []
    unknown_loader = torch.utils.data.DataLoader(unknown, batch_size, shuffle=True)
    model = getModel(dataset).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(unknown_loader):
            inputs = data[0].to(device)
            outputs = model(inputs)
            pred_unknown.append(outputs)
    # 예측값 불확실성 측정
    uncertainty = getEntropyBasedUncertainty(pred_unknown, n_class)
    uncertainty = [[i,u] for i, u in enumerate(uncertainty)]
    uncertainty = sorted(uncertainty, key = lambda x: -x[1])    # 불확실성 큰 순서대로 정렬
    
    chosen_data = np.array([unknown[i] for i, u in uncertainty[:k]])
    chosen_data = TensorDataset(chosen_data)
    return chosen_data
    
def choiceReverseTopk(dataset, k, unknown, n_class, batch_size):
    """
    dataset: known dataset
    k:unknown dataset 개수 설정한 값
    unknown: unknown 데이터
    n_class: unknown 클래스 번호
    """
    pred_unknown = []
    unknown_loader = torch.utils.data.DataLoader(unknown, batch_size, shuffle=True)
    model = getModel(dataset).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(unknown_loader):
            inputs = data[0].to(device)
            outputs = model(inputs)
            pred_unknown.append(outputs)
    # 예측값 불확실성 측정
    uncertainty = getEntropyBasedUncertainty(pred_unknown, n_class)
    uncertainty = [[i,u] for i, u in enumerate(uncertainty)]
    uncertainty = sorted(uncertainty, key = lambda x: x[1])    # 불확실성 작은 순서대로 정렬
    
    chosen_data = np.array([unknown[i] for i, u in uncertainty[:k]])
    chosen_data = TensorDataset(chosen_data)
    return chosen_data

def getHistogrambin(dataset, b, unknown, n_class, batch_size):
    """
    dataset: known dataset
    b: bin
    unknown: unknown 데이터
    n_class: unknown 클래스 번호
    """
    pred_unknown = []
    unknown_loader = torch.utils.data.DataLoader(unknown, batch_size, shuffle=True)
    model = getModel(dataset).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(unknown_loader):
            inputs = data[0].to(device)
            outputs = model(inputs)
            pred_unknown.append(outputs)
    # 예측값 불확실성 측정
    uncertainty = getEntropyBasedUncertainty(pred_unknown, n_class)
    uncertainty = [[i,u] for i, u in enumerate(uncertainty)]
    
    bin_data = [[] for _ in range(b)]
    for i, u in uncertainty:
        bin_data[int(b*u)].append(i)
        
    return bin_data
    
    
def choiceUniform(dataset, k, b, unknown, n_class, batch_size):
    """
    dataset: known dataset
    k:unknown dataset 개수 설정한 값
    b:
    unknown: unknown 데이터
    n_class: unknown 클래스 번호
    """
    bin_unknown = getHistogrambin(dataset, b, unknown, n_class, batch_size)
    p = [1/b]*b
    
    idx = []
    while len(idx) < k:
        i = np.random.choice(b, 1, p=p)[0]
        if len(bin_unknown[i]) > 0:
            d = np.random.choice(bin_unknown[i], 1)[0]
            if d not in idx:
                idx.append(d)
                
    chosen_data = np.array([unknown[i] for i in idx])
    chosen_data = TensorDataset(chosen_data)
    return chosen_data