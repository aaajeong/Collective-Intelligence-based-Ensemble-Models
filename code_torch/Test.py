import argparse
import os
import torch
import unknownClassification_Test

"""
1. Single Model Test


2. unknown classification Model Test
mnist: python Test.py --n_class 10
or 
       python Test.py --n_class 10 --selection topk
         
cifar10: python Test.py --dataset cifar10 --n_class 10
or
         python Test.py --dataset cifar10 --n_class 10 --selection topk
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--dataset', required=False, default='mnist')
    parser.add_argument('--n_class', type=int, required=True, help='number of classes excluding unknown class')
    parser.add_argument('--selection', required=False, default='random')
    
    args = parser.parse_args()
    
    # GPU Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')


    for num in [100, 200, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        print('======== num: '+str(num)+'========')
        
        model = unknownClassification_Test.TestModel(args.dataset, args.batch_size, args.n_class, args.selection, num)
    
    print('Test finished')