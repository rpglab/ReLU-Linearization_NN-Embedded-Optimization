# -*- coding: utf-8 -*-
"""
Neural network definition for the one-layer battery degradation model.

Authors:
    Cunzhi Zhao, Fan Jiang, and Xingpeng Li

If the code or data provided in this repository contributes to your work, we
kindly ask that you cite the following paper:
    Cunzhi Zhao, Fan Jiang, and Xingpeng Li, "Linearization of ReLU
    Activation Function for Neural Network-Embedded Optimization: Optimal
    Day-Ahead Energy Scheduling", Sustainable Energy, Grids and Networks,
    Aug. 2026.

Laboratory website:
    https://rpglab.github.io/
"""

import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        #self.dropout = nn.Dropout(p=0.1)
        # self.hidden1 = torch.nn.Linear(n_hidden, 10)
        # self.predict = torch.nn.Linear(10, n_output)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        
        
    def forward(self, x):
        x = nn.functional.relu(self.hidden(x))
        #x = self.dropout(x)
        # x = nn.functional.relu(self.hidden1(x))
        x = self.predict(x)
        
        return x


    
