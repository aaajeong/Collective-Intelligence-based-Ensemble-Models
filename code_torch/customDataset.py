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
        # print(len(self.targets2))   # 100000
        # print(self.targets2[50000:50010])
        
    def __getitem__(self):
        image = self.dataset[0]
        target = self.targets
        return (image, target)

    def __len__(self):
        return len(self.targets)