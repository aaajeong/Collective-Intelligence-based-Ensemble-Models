# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:42:25 2021

@author: onee
"""

import argparse

from numOfUnknown import trainModel as t1
from dataSelectionMethod import trainModel as t2

# python training.py --n_class 11 --n 0
# nohup python training.py --n_class 11 --n 0 &

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, required=False, default=300)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--dataset', required=False, default='mnist')
    parser.add_argument('--n_class', type=int, required=True, help='number of classes excluding unknown class')
    parser.add_argument('--n', type=int, required=True, help='Test Number') # 이게앙상블에 참여하는 모델 수  인가?

    args = parser.parse_args()

    # GPU setting
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    if args.n == 0:
        # Train pre-train model
        import numpy as np
        import model_struct
        from tensorflow.keras.utils import to_categorical

        if args.dataset == 'mnist':
            from keras.datasets import mnist
            (x_train, y_train), (_, _) = mnist.load_data()
        
        elif args.dataset == 'cifar10':
            import pickle
            with open('../data/cifar10.p','rb') as file:
                x_train = pickle.load(file)
                y_train = pickle.load(file)
        
        x_train = np.reshape(x_train,(len(x_train),28,28,1))
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = np.array(to_categorical(y_train,10))
        model = model_struct.MultiCls(x_train[0],10)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.fit(x_train,y_train,epochs=args.epochs,batch_size=args.batch_size)
        # 폴더 없으면 자동생성 하는 코드 추가
        if not os.path.isdir('../model/single'):
            os.mkdir('../model/single')   
        model.save('../model/single/'+args.dataset+'Network_epoch'+str(args.epochs)+'.h5')

    elif args.n == 1:   
        # Test1 training
        '''
        unknown 데이터의 개수에 따라 모델을 불러왔나보군
        for num in [500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
            t1(num,args.n_class,args.epochs,args.batch_size,args.dataset)

        '''
        # 잠깐...ㅎ
        from tmp_training import trainModel1 as tmp
        #for i in range(1,6):
        for i in range(4,6):    # 모델 이름 구별하기 위해서 넘버링
            for s in ['uniform']:
            #for s in ['random','topk','rtopk','uniform']:
                for n in [500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:  # unknown 데이터 개수에 따라 모델 학습
                #for n in [100,200,400,600,800,1000,1200,1400,1600,1800,2000]:
                    tmp(n,args.n_class,args.epochs,args.batch_size,args.dataset,i,s)

    elif args.n == 2:
        #Test2 training
        '''
        for s in ['random','topk','rtopk','uniform']:
            t2(args.dataset,args.n_class,s,args.epochs,args.batch_size)
        '''

        # 잠깐
        #from tmp_training import trainModel2 as tmp
        #for s in ['random','topk','rtopk','uniform']:
        #    for i in range(1,6):
        #        tmp(args.dataset,args.n_class,s,args.epochs,args.batch_size,i)

        # imbalanced 실험(불균형?)
        from tmp_training import trainModel_ib as tmp
        #for s in ['uniform']:
        for s in ['random','topk','rtopk','uniform']:
            for i in range(1,6):
                tmp(args.dataset,args.n_class,s,args.epochs,args.batch_size,i)

    #elif args.n == 3:
        # Test3 training
        # if you want to train own model, you can train the model in 'SingleModel.py'
        # you can ensemble With pre-saved models

     #elif args.n == 4:
         # Test3 training2
         # if you want to train own model, you can train the model in 'SingleModel.py'
         # you can ensemble With pre-saved models
