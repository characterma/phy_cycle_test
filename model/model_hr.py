import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net_gender_hr(nn.Module):
    def __init__(self):
        super(Net_gender_hr, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.bn1=nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(3, 2)

        self.conv3 = nn.Conv1d(32, 96, 3, 1)
        self.bn3 = nn.BatchNorm1d(96)
        self.max_pool3 = nn.MaxPool1d(3, 2)

        self.conv4 = nn.Conv1d(64, 96, 3, 1)
        self.bn4 = nn.BatchNorm1d(96)


        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Linear(96*1,1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.max_pool3(x)
        #
        # x = F.relu(self.bn4(self.conv4(x)))
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x
