import torch
from torch.utils.data import Dataset

class custom_wholeset(Dataset):
    r"""
    wholeset of a dataset using concatenation

    Arguments:
        datasets_list (Dataset): The Dataset list
        labels_list : targets as required for the datasets.
    """
    
    def __init__(self, dataset_list, labels_list):
        # self.dataset = torch.utils.data.Subset(dataset, indices)
        self.dataset = torch.utils.data.ConcatDataset(dataset_list)
        self.targets = torch.cat(labels_list, dim = 0)
        # self.dataset[0] : (이미지, 레이블)
        # self.dataset[0][0]: 채널 수
        # self.datset[0][0][0]: 32차원
        
    def __getitem__(self):
        # image = self.dataset[0]
        image = self.dataset
        target = self.targets
        return (image, target)

    def __len__(self):
        return len(self.targets)
    

class custom_unk_wholeset(Dataset):
    """
    Change the target of all unknown datasets to 'unknown class'

    Arguments:
        datasets_list (Dataset): The Dataset list
        k: unknwon dataset 개수 설정 값
        n_class: unknown 클래스 번호
    """
    
    def __init__(self, dataset_list, k, n_class):
        
        # labels 전체 unknown class로 처리
        for i in range(len(dataset_list)):
            unknown = dataset_list[i]
            
            for j in range(k):
                unknown.targets[j] = n_class
            
        
        self.dataset = torch.utils.data.ConcatDataset(dataset_list)
        
    def __getitem__(self):
        # image = self.dataset
        # target = self.targets
        final_unknown = self.dataset
        return final_unknown

    def __len__(self):
        return len(self.dataset)