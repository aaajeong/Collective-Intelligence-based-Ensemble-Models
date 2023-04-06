import math
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import RandomSampler
from torch.utils.data.dataset import random_split

def getModel(main_dataset):
    model_path = '../model/'+ main_dataset + 'Network_epoch300.h5'
    model = torch.load(model_path)
    return model

def checkData(dataset, selection, data_list):
    """
    dataset: mnist or cifar10
    selection: unknown 선택 기준 방법
    data_list: unknown 데이터를 랜덤 비복원 추출 한 인덱스 리스트
    """
    data_num = {'mnist':0, 'emnist':0, 'cifar10':0, 'cifar100':0, 'imagenet':0}
    
    
    
def getEntropyBasedUncertainty(data, n_class):
    uncertainty = []
    for d in data:
        tmp = 0
        for i in range(n_class):
            tmp = tmp + d[i]*math.log(d[i],n_class) if d[i] != 0 else tmp
        uncertainty.append(-1*tmp)
    return uncertainty


def choiceRandom(k, unknown):
    """
    k:unknown dataset 개수 설정한 값
    unknown: unknown 데이터(<class 'customDataset.custom_wholeset'>)
    n_class: unknown 클래스 번호
    """
    
    # k 만큼 분리
    # <torch.utils.data.dataset.Subset object at 0x7fba227f4610>
    chosen_data, _ = random_split(unknown, [k, len(unknown)-k])

    return chosen_data