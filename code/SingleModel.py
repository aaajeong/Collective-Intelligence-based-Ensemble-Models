# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:42:14 2021

@author: onee
"""

import os
import pickle
import argparse

import numpy as np
from tensorflow.keras.utils import to_categorical

import prepareData
import model_struct

def loadData(dataset, n_class, unknown):
    if dataset == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
    elif dataset == 'fashion':
        from keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    elif dataset == 'cifar10':
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    elif dataset == 'cifar100':
        from keras.datasets import cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        
    else:
        with open(dataset,'rb') as file:
            x_train = pickle.load(file)
            y_train = pickle.load(file)
            x_test = pickle.load(file)
            y_test = pickle.load(file)
            
    if unknown:
        if unknown == 'mnist':
            from keras.datasets import mnist
            (unknown_x_train, _), (unknown_x_test, _) = mnist.load_data() 
        elif unknown == 'fashion_mnist':
            from keras.datasets import fashion_mnist
            (unknown_x_train, _), (unknown_x_test, _) = fashion_mnist.load_data()
        else:
            with open(unknown,'rb') as file:
                unknown_x_train = pickle.load(file)
                _ = pickle.load(file)
                unknown_x_test = pickle.load(file)
                _ = pickle.load(file)
                
        num = 5000
        # num = 500
        selection = 'uniform'
        data_name = os.path.split(dataset)[-1].replace('.p','')
        x_train, y_train = prepareData.unknownClassData(data_name,n_class,x_train,y_train,unknown_x_train,num,selection)
        x_test, y_test = prepareData.unknownClassData(data_name,n_class,x_test,y_test,unknown_x_test,num,selection)
        
    if not unknown:
        if len(x_train[0].shape) == 2:
            x_train = np.reshape(x_train,(len(x_train),28,28,1))
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = np.array(to_categorical(y_train,n_class))
        
        if len(x_test[0].shape) == 2:
            x_test = np.reshape(x_test,(len(x_test),28,28,1))
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = np.array(to_categorical(y_test,n_class))
    
        import random
        d_train = list(zip(x_train, y_train))
        random.shuffle(d_train)
        x_train, y_train = zip(*d_train)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
    
    return x_train, y_train, x_test, y_test

def trainSingleModel(datapath, unknown, n_class, epochs, batch_size):    
    x_train, y_train, x_test, y_test = loadData(datapath,n_class,unknown)
    
    if unknown:
        n_class += 1
    
    #model = model_struct.MultiCls(x_train[0],n_class) # you can add model structure in 'model_struct.py'
    model = model_struct.MultiCls_color(x_train[0], n_class)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
    
    data_name = os.path.split(datapath)[-1].replace('.p','')
    if unknown:
        data_name += '_unknown'
    if not os.path.isdir('../model/single_model'):
        os.mkdir('../model/single_model')
    model.save('../model/single_model/single_model_'+data_name+'_epochs'+str(epochs)+'.h5')

def trainSingleModel_tmp(datapath, unknown, n_class, epochs, batch_size, i):
    x_train, y_train, x_test, y_test = loadData(datapath,n_class,unknown)
    
    if unknown:
        n_class += 1

    model = model_struct.MultiCls(x_train[0],n_class) # you can add model structure in 'model_struct.py'
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)

    data_name = os.path.split(datapath)[-1].replace('.p','')
    if unknown:
        data_name += '_unknown'
    if not os.path.isdir('../model/test3/'+str(i)):
        os.mkdir('../model/test3/'+str(i))
    model.save('../model/test3/'+str(i)+'/single_model_'+data_name+'_epochs'+str(epochs)+'.h5')

# total dataset은 mnist, fashion, cifar10 순으로

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, required=False, default=300)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--datapath', required=True)
    parser.add_argument('--unknownpath', required=False)
    parser.add_argument('--n_class', type=int, required=True, help='number of classes')
    
    args = parser.parse_args()

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    #trainSingleModel(args.datapath,args.unknownpath,args.n_class,args.epochs,args.batch_size)

    #for n_class, d in [[20,'./data/cifar100_sub1.p'],[20,'./data/cifar100_sub2.p'],[20,'./data/cifar100_sub3.p']]:
        #trainSingleModel_tmp(d,'',n_class,args.epochs,args.batch_size,0)

    # 잠시 빠른 학습용
    # for n_class, d, unknown in [[5,'./data/total_coarse.p','']]:
    for n_class, d, unknown in [[30,'../data/total_mandfandc.p','../data/emnist.p'],[15,'../data/mnist_and_fandc.p','../data/emnist.p'],[15,'../data/fashion_and_mandc.p','../data/emnist.p'],[15,'../data/cifar_and_mandf.p','../data/emnist.p']]:
    #for n_class, d, unknown in [[45,'./data/cifar100_sub_total.p',''],[20,'./data/cifar100_sub1.p','./data/cifar100_sub_unknown.p'],[20,'./data/cifar100_sub2.p','./data/cifar100_sub_unknown.p'],[20,'./data/cifar100_sub3.p','./data/cifar100_sub_unknown.p']]: # [29,'./data/cifar100_sub_total.p','']
    # for d, unknown in [['mnist','./data/mnist_unknown.p'],['fashion','./data/fashion_unknown.p'],['./data/cifar10.p','./data/cifar10_unknown.p']]:
        for i in range(1,6):
            trainSingleModel_tmp(d,unknown,n_class,args.epochs,args.batch_size,i)
    #for i in range(1,6):
    #    trainSingleModel_tmp(args.datapath,args.unknownpath,args.n_class,args.epochs,args.batch_size,i)
