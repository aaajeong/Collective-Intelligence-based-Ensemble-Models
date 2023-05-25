import torch
import selectData
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# =========================================================== #
# 기준에 따라서 unknwon 데이터를 선별하고 데이터를 준비하는 작업이다.
# =========================================================== #

def labelTounknown(data_list, n_class):
    """
    Change the target of all unknown datasets to 'unknown class'

    Arguments:
        datasets_list (Dataset): The Dataset list
        n_class: unknown 클래스 번호
    """
    for i in range(len(data_list)):
        final_list = []
        unknown = data_list[i]
        
        for j in range(len(unknown)):
            unknown.targets[j] = n_class
        final_list.append(unknown)
    return final_list
    # return data_list

def unknownClassData(main_dataset, trainset, n_class, unknown, num, selection, batch_size):
    """
    main_dataset: mnist or cifar10
    trainst: main_dataset에 해당하는 학습 데이터셋
    n_class: 총 클래스 개수 + 1
    unknwon: unknwon 설정된 데이터
    num: unknown 설정할 데이터 개수
    selection: unknwon 데이터 선별 기준
    """
    
    """
    해야할것
    원래 trainset에 unknown 데이터를 붙여야 함. (인풋값, 레이블 값 둘다.)
    """
    # <class 'torch.utils.data.dataset.Subset'>
    unknown = choiceUnknown(main_dataset, unknown, num, selection, n_class, batch_size)  
    
    # 최종 known + unknown 합치기
    final_trainset = torch.utils.data.ConcatDataset([trainset, unknown])
    
    return final_trainset


def choiceUnknown(main_dataset, unknown, num, selection, n_class, batch_size):
    """
    main_dataset: mnist or cifar10
    unknown: unknown 데이터
    num:unknown dataset 개수 설정한 값
    selection: unknown 데이터 선별 기준
    batch_size: unknown 데이터 batch size
    """
    
    if selection == 'random':
        unknown = selectData.choiceRandom(num,unknown)
    elif selection == 'uniform':
        b = 20 # 이거 몇으로 설정할지 고민하자!
        unknown = selectData.choiceUniform(main_dataset,num,b,unknown,n_class, batch_size)
    elif selection == 'topk':
        unknown = selectData.choiceTopk(main_dataset,num,unknown,n_class, batch_size)
    elif selection == 'rtopk':
        unknown = selectData.choiceReverseTopk(main_dataset,num,unknown,n_class, batch_size)
        
    return unknown