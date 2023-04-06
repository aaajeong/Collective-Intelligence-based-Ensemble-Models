# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:35:43 2021

@author: SM-PC
"""
# 기준 dataset에 따라 unknown data를 선별하고 이를 학습할 수 있게 만든 코드
import os
import pickle
import numpy as np

import prepareData
import model_struct

def trainModel(dataset, n_class, selection, epochs, batch_size):
    # load data for train
    if dataset == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (_, _) = mnist.load_data()
        # print(x_train.shape)    #(60000, 28, 28)
        
        # with open('./data/cifar10.p','rb') as file:
        from keras.datasets import cifar10
        (unknown, _), (_, _) = cifar10.load_data()
        # print(unknown.shape)    # (50000, 32, 32, 3)
        # unknown = pickle.load(file)
        # _ = pickle.load(file)
        # _ = pickle.load(file)
        # _ = pickle.load(file)
        
    elif dataset == 'cifar10':
        with open('./data/cifar10.p','rb') as file:
            x_train = pickle.load(file)
            y_train = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            
        from keras.datasets import mnist
        (unknown, _), (_, _) = mnist.load_data()
    
    # with open('./data/emnist.p','rb') as file:
    #     emnist = pickle.load(file)
    #     _ = pickle.load(file) 
    # from keras.datasets import fashion_mnist
    # (fashion_mnist_data, _), (_, _) = fashion_mnist.load_data()
    
    from keras.datasets import cifar100
    (cifar100_data, _), (_, _) = cifar100.load_data()
    
    # with open('./data/imagenet.p','rb') as file:
    #     imagenet = pickle.load(file)
    #     _ = pickle.load(file)
    #     _ = pickle.load(file)
    #     _ = pickle.load(file)
    
    #unknown = np.vstack([unknown,emnist,imagenet])
    # unknown = np.vstack([emnist,imagenet])
    unknown = np.vstack([unknown, cifar100_data])   
    # print(len(unknown)) #100000개
        
    num = 5000  # unknown 데이터 얼만큼 할지 선택하는 것?
    x_train, y_train = prepareData.unknownClassData(dataset,n_class,x_train,y_train,unknown,num,selection)

    model = model_struct.MultiCls(x_train[0],n_class+1)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
    
    if not os.path.isdir('../model/test'):
        os.mkdir('../model/test')
    model.save('../model/test/'+dataset+'_dataselection_'+selection+'.h5')

# 코드 테스트
# selection이 topk, reversetopk, histogram, uniform은 get_model 함수가 있어서 사전학습 모델이 존재하야함.
# trainModel('mnist', 11, 'uniform', 300, 128)
trainModel('mnist', 11, 'random', 300, 128)