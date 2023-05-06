import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

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
    


class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        tmpset, _ = random_split(dataset, [indices, len(dataset)-indices])
        self.dataset = tmpset
        self.targets = tmpset.indices
        
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        targets = self.targets[idx]
        return (image, targets)

    def __len__(self):
        return len(self.dataset)


# transform_3ch = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# unk_trainset = torchvision.datasets.ImageNet(root='../data/imagenet', split = 'train',
#                                                      download=None, transform=transform_3ch)

# subsetdatset = custom_subset(unk_trainset, 50000)