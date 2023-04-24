import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 합성곱 연산 (입력 채널수 3, 출력 채널수 6, 필터크기 5x5 , stride=1(defualt))
        self.pool1 = nn.MaxPool2d(2, 2) # 합성곱 연산 (필터크기 2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 합성곱 연산 (입력 채널수 6, 출력 채널수 16, 필터크기 5x5 , stride=1(defualt))
        self.pool2 = nn.MaxPool2d(2, 2) # 합성곱 연산 (필터크기 2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 5x5 피쳐맵 16개를 일렬로 피면 16*5*5개의 노드가 생성
        
        self.fc2 = nn.Linear(120, n_class+1) # 120개 노드에서 클래스의 개수인 (n_class + 1)개의 노드로 연산 

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) # conv1 -> ReLU -> pool1
        x = self.pool2(F.relu(self.conv2(x))) # conv2 -> ReLU -> pool2
        x = x.view(-1, 16 * 5 * 5) # 5x5 피쳐맵 16개를 일렬로 만든다.
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)

        return x


class MultiCls(nn.Module):
    def __init__(self, n_class):
        super(MultiCls, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(64))
        
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        
        self.fc2 = nn.Linear(128, n_class+1) # 128개 노드에서 클래스의 개수인 (n_class + 1)개의 노드로 연산 

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1) # flatten
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class CNN(torch.nn.Module):

    def __init__(self, n_class):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 32, 32, 1)
        #    Conv     -> (?, 32, 32, 32)
        #    Pool     -> (?, 16, 16, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 16, 16, 32)
        #    Conv      ->(?, 16, 16, 64)
        #    Pool      ->(?, 8, 8, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 8x8x64 inputs -> 10 outputs
        self.fc1 = torch.nn.Linear(8 * 8 * 64, 128, bias=True)
        self.fc2 = nn.Linear(128, n_class+1) # 128개 노드에서 클래스의 개수인 (n_class + 1)개의 노드로 연산 

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc1(out)
        out = self.fc2(out)
        return out