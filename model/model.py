import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# data = np.load('my_data/image_512', allow_pickle=True)
# label=np.load('my_data/label_512', allow_pickle=True)
# pre = np.array(label)
# a = np.sum(pre == 0)
# b = np.sum(pre == 1)
# c = np.sum(pre == 2)
# d = np.sum(pre == 3)
# print(a,b,c,d)
# MAX=data.max()
# MIN=data.min()
# MEAN=data.mean()
# STD=data.std()
class Net_phy(nn.Module):
    def __init__(self):
        super(Net_phy, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2,padding=3)
        self.bn1=nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.conv3 = nn.Conv1d(16, 32, 5, 2)
        self.bn2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(3, 2)

        self.conv5 = nn.Conv1d(32, 64, 5, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(3, 2)

        self.conv7 = nn.Conv1d(64, 96, 3, 1)
        self.bn4 = nn.BatchNorm1d(96)
        self.max_pool4 = nn.MaxPool1d(3, 2)

        self.conv9 = nn.Conv1d(96, 96, 3, 1)
        self.bn5 = nn.BatchNorm1d(96)
        self.avg=nn.AdaptiveAvgPool1d(1)
        self.fc3=nn.Linear(96 * 1,4)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)

        x = F.relu(self.bn2(self.conv3(x)))
        x = self.max_pool2(x)

        x = F.relu(self.bn3(self.conv5(x)))
        x = self.max_pool3(x)

        x = F.relu(self.bn4(self.conv7(x)))
        x = self.max_pool4(x)

        x = F.relu(self.bn5(self.conv9(x)))

        x=self.avg(x)
        x=torch.flatten(x,1)
        x = self.fc3(x)

        return x
# from torchsummary import summary
# device=torch.device("cuda")
# net=Net_phy().to(device)
# summary(net,input_size=(1,512))