import torch
import selectData
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# =========================================================== #
# 기준에 따라서 unknwon 데이터를 선별하고 데이터를 준비하는 작업이다.
# =========================================================== #

    
def choiceUnknown(main_dataset, unknown, num, selection, n_class):
    """
    main_dataset: mnist or cifar10
    unknown: unknown 데이터
    num:unknown dataset 개수 설정한 값
    selection: unknown 데이터 선별 기준
    """
    # n_class = 10
    
    if selection == 'random':
        unknown = selectData.choiceRandom(main_dataset,num,unknown, n_class)
    elif selection == 'uniform':
        b = 20 # 이거 몇으로 설정할지 고민하자!
        unknown = selectData.choiceUniform(main_dataset,num,b,unknown,n_class)
    elif selection == 'topk':
        unknown = selectData.choiceTopk(main_dataset,num,unknown,n_class)
    elif selection == 'rtopk':
        unknown = selectData.choiceReverseTopk(main_dataset,num,unknown,n_class)
        
    return unknown

def unknownClassData(main_dataset, trainset, n_class, unknown, num, selection):
    """
    main_dataset: mnist or cifar10
    trainst: main_dataset에 해당하는 학습 데이터셋
    n_class: 총 클래스 개수 + 1
    unknwon: unknwown으로 설정된 데이터
    num: unknown 설정할 데이터 개수
    selection: unknwon 데이터 선별 기준
    """
    
    """
    해야할것
    원래 trainset에 unknown 데이터를 붙여야 함. (인풋값, 레이블 값 둘다.)
    """
    # <class 'torch.utils.data.dataset.Subset'>
    unknown = choiceUnknown(main_dataset, unknown, num, selection, n_class)  
    
    # 최종 known + unknown 합치기
    final_trainset = torch.utils.data.ConcatDataset([trainset, unknown])
    final_trainloader = torch.utils.data.DataLoader(final_trainset, shuffle=True) 
    # print(type(final_trainloader))
    # print('?>')
    dataiter = iter(final_trainloader)
    # unknown이랑 known이랑 데이터 비교해보면서, 데이터 형식 동일한지 확인하기
    # 데이터 로더가 안되는 중임.
    images, labels = next(dataiter) 
    print(labels.size())
    print('?>')
    # return final_trainset