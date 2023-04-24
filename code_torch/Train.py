import argparse
import os
import torch
import singleModel
import unknownClassification

"""
1. Single Model 학습
python Train.py --n_class 9 --n 0
nohup python Train.py --n_class 9 --n 0 &

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
        model = unknownClassification.trainModel(args.dataset, args.n_class, args.selection, args.epochs, args.batch_size)
    else:
        print('None')
        
        