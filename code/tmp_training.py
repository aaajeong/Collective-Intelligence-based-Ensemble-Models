# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:34:11 2021

@author: onee
"""

import os
import pickle
import numpy as np

import prepareData
import model_struct
# trainModel1: numofUnknown.py와 비슷
# trainModel2: dataSelectionMethod.py와 비슷
# trainModel_ib: 데이터 불균형 실험인가?
def trainModel1(num, n_class, epochs, batch_size, dataset, cnt, selection):
    # load data for train
    if dataset == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (_, _) = mnist.load_data()
        
        #with open('./data/mnist_1000.p','rb') as file:
        #    x_train = pickle.load(file)
        #    y_train = pickle.load(file)

        with open('../data/cifar10.p','rb') as file:
            unknown = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)

            # 2만5천개 뽑기로 하자
            idx_rand = np.random.choice(len(unknown), 25000, replace=False)
            unknown = np.array([unknown[i] for i in idx_rand])

    elif dataset == 'cifar10':
        with open('../data/cifar10.p','rb') as file:
        #with open('./data/cifar10_1000.p','rb') as file:
            x_train = pickle.load(file)
            y_train = pickle.load(file)
            #_ = pickle.load(file)
            #_ = pickle.load(file)

        from keras.datasets import mnist
        (unknown, _), (_, _) = mnist.load_data()

        # 2만5천개 뽑기로 하자
        idx_rand = np.random.choice(len(unknown), 25000, replace=False)
        unknown = np.array([unknown[i] for i in idx_rand])
    
    with open('../data/emnist.p','rb') as file:
        emnist = pickle.load(file)
        _ = pickle.load(file)   
    with open('../data/imagenet.p','rb') as file:
        imagenet = pickle.load(file)
        _ = pickle.load(file)
        _ = pickle.load(file)
        _ = pickle.load(file)
    
    # 2만5천개 뽑기로 하자
    idx_rand = np.random.choice(len(emnist), 25000, replace=False)
    emnist = np.array([emnist[i] for i in idx_rand])
    idx_rand = np.random.choice(len(imagenet), 25000, replace=False)
    imagenet = np.array([imagenet[i] for i in idx_rand])

    unknown = np.vstack([unknown,emnist,imagenet])
    
    x_train, y_train = prepareData.unknownClassData(dataset,n_class,x_train,y_train,unknown,num,selection)
    
    model = model_struct.MultiCls(x_train[0],n_class+1)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)

    if not os.path.isdir('../model/test1/'+str(cnt)):
        os.mkdir('../model/test1/'+str(cnt))

    model.save('../model/test1/'+str(cnt)+'/'+dataset+'_unknownNetwork_'+selection+'_'+str(num)+'.h5')

def trainModel2(dataset, n_class, selection, epochs, batch_size, cnt):
    # load data for train
    if dataset == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (_, _) = mnist.load_data()

        with open('../data/cifar10.p','rb') as file:
            unknown = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)

        # 2만5천개 뽑기로 하자
        idx_rand = np.random.choice(len(unknown), 25000, replace=False)
        unknown = np.array([unknown[i] for i in idx_rand])

    elif dataset == 'cifar10':
        with open('../data/cifar10.p','rb') as file:
            x_train = pickle.load(file)
            y_train = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)

        from keras.datasets import mnist
        (unknown, _), (_, _) = mnist.load_data()

        # 2만5천개 뽑기로 하자
        idx_rand = np.random.choice(len(unknown), 25000, replace=False)
        unknown = np.array([unknown[i] for i in idx_rand])

    with open('../data/emnist.p','rb') as file:
        emnist = pickle.load(file)
        _ = pickle.load(file)
    with open('../data/imagenet.p','rb') as file:
        imagenet = pickle.load(file)
        _ = pickle.load(file)
        _ = pickle.load(file)
        _ = pickle.load(file)

    # 2만5천개 뽑기로 하자
    idx_rand = np.random.choice(len(emnist), 25000, replace=False)
    emnist = np.array([emnist[i] for i in idx_rand])
    idx_rand = np.random.choice(len(imagenet), 25000, replace=False)
    imagenet = np.array([imagenet[i] for i in idx_rand])

    #unknown = np.vstack([emnist,imagenet])
    unknown = np.vstack([unknown,emnist,imagenet])

    num = 5000 # unknown data를 5000개 뽑는다?
    x_train, y_train = prepareData.unknownClassData(dataset,n_class,x_train,y_train,unknown,num,selection)

    model = model_struct.MultiCls(x_train[0],n_class+1)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)

    if not os.path.isdir('../model/test2/'+str(cnt)):
        os.mkdir('../model/test2/'+str(cnt))

    model.save('../model/test2/'+str(cnt)+'/'+dataset+'_dataselection_'+selection+'.h5')

def trainModel_ib(dataset, n_class, selection, epochs, batch_size, cnt):
    # load data for train
    with open('../data/unknown_data_5000.p','rb') as file:
        mnist = pickle.load(file)
        emnist = pickle.load(file)
        cifar10 = pickle.load(file)
        imagenet = pickle.load(file)

    if dataset == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (_, _) = mnist.load_data()

        with open('../data/imagenet.p','rb') as file:
            imagenet = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)

        unknown = np.vstack([cifar10,emnist,imagenet])

    elif dataset == 'cifar10':
        with open('../data/cifar10.p','rb') as file:
            x_train = pickle.load(file)
            y_train = pickle.load(file)
            _ = pickle.load(file)
            _ = pickle.load(file)

        with open('../data/emnist.p','rb') as file:
            emnist = pickle.load(file)
            _ = pickle.load(file)

        unknown = np.vstack([mnist,emnist,imagenet])

    num = 5000
    x_train, y_train = prepareData.unknownClassData(dataset,n_class,x_train,y_train,unknown,num,selection)

    model = model_struct.MultiCls(x_train[0],n_class+1)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
    
    if not os.path.isdir('../model/test2/'+str(cnt)):
        os.mkdir('../model/test2/'+str(cnt))
        
    model.save('../model/test2/'+str(cnt)+'/'+dataset+'_imbalance_dataselection_'+selection+'.h5')
