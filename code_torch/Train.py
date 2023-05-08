import argparse
import os
import torch
import singleModel
import unknownClassification

"""
1. Single Model 학습
mnist: python Train.py --n_class 9 --n 0
cifar10: python Train.py --dataset cifar10 --n_class 9 --n 0
nohup python Train.py --n_class 9 --n 0 &

2. unknown class 학습
mnist: python Train.py --n_class 10 --n 1
cifar10: python Train.py --dataset cifar10 -n_class 10 --n 1
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, required=False, default=300)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--dataset', required=False, default='mnist')
    parser.add_argument('--n_class', type=int, required=True, help='number of classes excluding unknown class')
    parser.add_argument('--selection', required=False, default='random')
    parser.add_argument('--n', type=int, required=True, help='Test Number') # 이게앙상블에 참여하는 모델 수  인가?
    
    args = parser.parse_args()
    
    # GPU Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')

    # PATH = '../model/test/mnist_dataselection_random.h5'
    
    
    if args.n == 0:
        # Train original single model: no unknown.
        model = singleModel.trainSingleModel(args.dataset, args.epochs, args.batch_size, args.n_class)
        
        if not os.path.isdir('../model/single'):
            os.mkdir('../model/single')   
        torch.save(model, '../model/single/'+args.dataset+'_epoch'+str(args.epochs)+'.h5')
    elif args.n == 1:
        for num in [100, 200, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            print('======== num: '+str(num)+'========')
            model = unknownClassification.trainModel(args.dataset, args.n_class, args.selection, args.epochs, args.batch_size, num)
        
        print('Train finished')
    else:
        print('None')
        
        