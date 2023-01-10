import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MIN = np.array([2.31816900e+00, 1.83594000e-01, 1.58203000e-01, 3.67188000e-01,
                7.42190000e-02, 1.83170000e-01, 1.03598000e-01, 4.37750000e-02,
                2.53290000e-02, 1.31066980e+01, 2.74196930e+01, 4.71136261e+02])

MAX = np.array([9.62324520e+01, 9.58984000e-01, 8.33984000e-01, 1.54882800e+00,
                5.87891000e-01, 8.96402000e-01, 8.16830000e-01, 7.41067000e-01,
                8.40605000e-01, 1.27976133e+04, 1.92170703e+04, 8.09979531e+04])


class Net_gender_ppg(nn.Module):
    def __init__(self, n_input, n_hidden, out_hidden, n_output):
        super(Net_gender_ppg, self).__init__()
        self.hidden1 = nn.Linear(n_input, 256)
        self.hidden2 = nn.Linear(256, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, 1024)
        self.hidden4 = nn.Linear(1024, 512)
        self.hidden5 = nn.Linear(512, out_hidden)
        self.predict = nn.Linear(out_hidden, n_output)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input):
        out = F.relu(self.hidden1(input))
        out = F.relu(self.dropout(self.hidden2(out)))
        out = F.relu(self.dropout(self.hidden3(out)))
        out = F.relu(self.dropout(self.hidden4(out)))
        out = F.relu(self.dropout(self.hidden5(out)))
        out = torch.sigmoid(self.predict(out))
        return out
