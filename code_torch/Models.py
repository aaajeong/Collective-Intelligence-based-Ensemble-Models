import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 합성곱 연산 (입력 채널수 3, 출력 채널수 6, 필터크기 5x5 , stride=1(defualt))
        self.pool1 = nn.MaxPool2d(2, 2) # 합성곱 연산 (필터크기 2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 합성곱 연산 (입력 채널수 6, 출력 채널수 16, 필터크기 5x5 , stride=1(defualt))
        self.pool2 = nn.MaxPool2d(2, 2) # 합성곱 연산 (필터크기 2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 5x5 피쳐맵 16개를 일렬로 피면 16*5*5개의 노드가 생성
        # self.fc2 = nn.Linear(120, 10) # 120개 노드에서 클래스의 개수인 10개의 노드로 연산
        
        self.fc2 = nn.Linear(120, 10) # 120개 노드에서 클래스의 개수인 10개의 노드로 연산

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) # conv1 -> ReLU -> pool1
        x = self.pool2(F.relu(self.conv2(x))) # conv2 -> ReLU -> pool2
        x = x.view(-1, 16 * 5 * 5) # 5x5 피쳐맵 16개를 일렬로 만든다.
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)

        return x
    

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 합성곱 연산 (입력 채널수 3, 출력 채널수 6, 필터크기 5x5 , stride=1(defualt))
        self.pool1 = nn.MaxPool2d(2, 2) # 합성곱 연산 (필터크기 2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 합성곱 연산 (입력 채널수 6, 출력 채널수 16, 필터크기 5x5 , stride=1(defualt))
        self.pool2 = nn.MaxPool2d(2, 2) # 합성곱 연산 (필터크기 2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 5x5 피쳐맵 16개를 일렬로 피면 16*5*5개의 노드가 생성
        
        self.fc2 = nn.Linear(120, 1) # 120개 노드에서 클래스의 개수인 10개의 노드로 연산

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) # conv1 -> ReLU -> pool1
        x = self.pool2(F.relu(self.conv2(x))) # conv2 -> ReLU -> pool2
        x = x.view(-1, 16 * 5 * 5) # 5x5 피쳐맵 16개를 일렬로 만든다.
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)

        return x