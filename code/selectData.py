# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:49:46 2021

@author: onee
"""

import math
import numpy as np

from keras.models import load_model

def getModel(dataset):
    model_path = './model/'+dataset+'Network_epoch300.h5'
    model = load_model(model_path)
    return model

# checkData(dataset,'Random',[i for i in idx_rand])

def checkData(dataset, selection, data_list):
    """
    dataset: mnist or cifar10
    selection: unknown 선택 기준 방법
    data_list: unknown 데이터를 랜덤 비복원 추출 한 인덱스 리스트
    """
    n_data = {'mnist':0, 'emnist':0, 'cifar10':0, 'cifar100':0, 'imagenet':0}
    
    n_data[dataset] = 50000
    if dataset == 'mnist':
        n_data[dataset] += 10000

    # for d in data_list:
    #     if d < 25000:
    #         if dataset == 'mnist':
    #             n_data['cifar10'] += 1
    #         elif dataset == 'cifar10':
    #             n_data['mnist'] += 1
    #     elif 25000 <= d < 50000:
    #         n_data['emnist'] += 1
    #     elif d >= 50000:
    #         n_data['imagenet'] += 1
    
    for d in data_list:
        if d < 50000:
            if dataset == 'mnist':
                n_data['cifar10'] += 1
            elif dataset == 'cifar10':
                n_data['mnist'] += 1
        elif d >= 50000:
            n_data['cifar100'] += 1
            
    '''
    if dataset == 'mnist':
        n_data[dataset] = 60000
        for d in data_list:
            if d < 50000:
                n_data['cifar10'] += 1
            elif 50000 <= d < 100000:
                n_data['emnist'] += 1
            else:
                n_data['imagenet'] += 1
    elif dataset =='cifar10':
        n_data[dataset] = 50000
        for d in data_list:
            if d < 60000:
                n_data['mnist'] += 1
            elif 60000 <= d < 110000:
                n_data['emnist'] += 1
            else:
                n_data['imagenet'] += 1
    '''

    # for d in data_list:
    #     if d < 50000:
    #         n_data['emnist'] += 1
    #     else:
    #         n_data['imagenet'] += 1

    import os
    if not os.path.isdir('../checkData'):
        os.mkdir('../checkData')
        
    if not os.path.isfile('../checkData/'+dataset+'_'+selection+'_checkdata.txt'):
        f = open(r'../checkData/'+dataset+'_'+selection+'_checkdata.txt','w')
    else:
        f = open(r'../checkData/'+dataset+'_'+selection+'_checkdata.txt','a')
    f.write('====='+str(len(data_list))+'=====\n')
    for k, v in n_data.items():
        f.write(k+': '+str(v)+'\n')
    f.write('\n\n')
    f.close()

def getEntropyBasedUncertainty(data, n_class):
    uncertainty = []
    for d in data:
        tmp = 0
        for i in range(n_class):
            tmp = tmp + d[i]*math.log(d[i],n_class) if d[i] != 0 else tmp
        uncertainty.append(-1*tmp)
    return uncertainty

def choiceTopk(dataset, k, unknown, n_class):
    model = getModel(dataset)
    pred_unknown = model.predict(unknown)
    uncertainty = getEntropyBasedUncertainty(pred_unknown, n_class)
    
    uncertainty = [[i,u] for i, u in enumerate(uncertainty)]
    uncertainty = sorted(uncertainty, key = lambda x: -x[1])

    c_data = np.array([unknown[i] for i, u in uncertainty[:k]])
    
    # check
    checkData(dataset,'Top-k',[i for i, u in uncertainty[:k]])

    return c_data

def choiceReverseTopk(dataset, k, unknown, n_class):
    model = getModel(dataset)
    pred_unknown = model.predict(unknown)
    uncertainty = getEntropyBasedUncertainty(pred_unknown, n_class)
    
    uncertainty = [[i,u] for i, u in enumerate(uncertainty)]
    uncertainty = sorted(uncertainty, key = lambda x: x[1])

    c_data = np.array([unknown[i] for i, u in uncertainty[:k]])
    
    # check
    checkData(dataset,'Reverse_Top-k',[i for i, u in uncertainty[:k]])

    return c_data


def MakeHistogram(b, bin_data):   
    h = [len(i) for i in bin_data]
    
    n_data = sum(h)
    h_star = [(1/b)*n_data]*b
    
    p = []
    p_sum = 0
    for i in range(b):
        c = 0.001
        tmp_p = max(h_star[i]-h[i],c)
        p_sum += tmp_p
        p.append(tmp_p)
    p = [i/p_sum for i in p]
    
    return p

def getHistogrambin(dataset, b, data, n_class):
    model = getModel(dataset)
    pred_data = model.predict(data)
    uncertainty = getEntropyBasedUncertainty(pred_data, n_class)
    uncertainty = [[i,u] for i, u in enumerate(uncertainty)]
    
    bin_data = [[] for _ in range(b)]
    for i, u in uncertainty:
        bin_data[int(b*u)].append(i)
        
    return bin_data

def choiceUniform(dataset, k, b, data, unknown, n_class):
    #bin_data = getHistogrambin(dataset,b,data,n_class)
    #p = MakeHistogram(b,bin_data)
    
    bin_unknown = getHistogrambin(dataset,b,unknown,n_class)
    #p = MakeHistogram(b,bin_unknown)
    p = [1/b]*b

    idx = []
    while len(idx) < k:
        i = np.random.choice(b, 1, p=p)[0]
        if len(bin_unknown[i]) > 0:
            d = np.random.choice(bin_unknown[i], 1)[0]
            if d not in idx:
                idx.append(d)
                
                # for update probability
                # bin_data[i].append(d)
                # p = MakeHistogram(b,bin_data)
    c_data = np.array([unknown[i] for i in idx])
    
    # check 
    checkData(dataset,'Uniform',[i for i in idx])

    return c_data

def choiceRandom(main_dataset, k, data):
    """
    main_dataset: mnist or cifar10
    k:unknown dataset 개수 설정한 값
    data: unknown 데이터
    """
    
    idx_rand = np.random.choice(len(data), k, replace=False)    # len(data) 중 k개를 비복원 추출 -> 인덱스 넘버가 추출된다.
    print('idx_rand: ', idx_rand)
    
    chosen_data = np.array([data[i] for i in idx_rand])  # 인덱스 넘버로 uknown 데이터 다시 가져온다.

    # check
    checkData(main_dataset,'Random',[i for i in idx_rand])

    return chosen_data
