# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:34:25 2021

@author: onee
"""

import selectData
import numpy as np

from tensorflow.keras.utils import to_categorical

def choiceUnknown(dataset, data, unknown, num, selection):
    n_class = 10
    
    if selection == 'random':
        unknown = selectData.choiceRandom(dataset,num,unknown)
    elif selection == 'uniform':
        b = 20 # 이거 몇으로 설정할지 고민하자!
        unknown = selectData.choiceUniform(dataset,num,b,data,unknown,n_class)
    elif selection == 'topk':
        unknown = selectData.choiceTopk(dataset,num,unknown,n_class)
    elif selection == 'rtopk':
        unknown = selectData.choiceReverseTopk(dataset,num,unknown,n_class)
        
    return unknown

def unknownClassData(dataset, n_class, data, labels, unknown, num, selection):
    """
    dataset: mnist or cifar10
    n_class:
    data: x_train
    labels: y_train
    unknwon: unknwown으로 설정된 데이터
    num: unknown 설정할 데이터 개수
    selection: unknwon 데이터 선별 기준
    """
    
    if len(data[0].shape) == 2:
        
        data = np.reshape(data,(len(data),28,28,1)) 
        # unknown = np.reshape(unknown,(len(unknown),28,28,1))
        unknown = np.reshape(unknown,(len(unknown),32,32,3))

    unknown = choiceUnknown(dataset,data,unknown,num,selection)
    
    print('unknown.shape: ', unknown.shape) # (5000, 32, 32, 3)
    print('data.shape: ', data.shape) # 
    data = np.vstack([data,unknown])    # 학습데이터에 unknown을 다시 쌓는다.
    labels = np.hstack([labels,np.array([n_class]*len(unknown))])
    
    import random
    r_data = list(zip(data, labels))
    random.shuffle(r_data)
    data, labels = zip(*r_data)
    data = np.array(data)
    labels = np.array(labels)
    
    data = data.astype('float32')
    data /= 255
    
    labels = np.array(to_categorical(labels,n_class+1))
        
    return data, labels

# 구현이 덜 된 것 같은데?
def hardsharingData(data, labels, unknown, num, selection):
    unknown = choiceUnknown(data,unknown,num,selection)
    return data, labels
