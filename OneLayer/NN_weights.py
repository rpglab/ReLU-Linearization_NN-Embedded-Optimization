# -*- coding: utf-8 -*-
"""
Weight and bias extraction utility for the one-layer battery degradation model.

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

from pyomo.environ import *
import pandas as pd
#import gurobipy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time

load_rnn = torch.load('trained_DNN.pt')
#input data 
Capacity = 0.9
Temp = 25 / 35
DISC = 85/300/2
SOC = 38 / 100
DOD = 0.28
Type = 1
#input array
# input_x1 = np.zeros((6,1))
input_x1 = np.zeros((5,1))
input_x1[0,0] = Capacity
input_x1[1,0] = Temp
input_x1[2,0] = DISC
input_x1[3,0] = SOC
input_x1[4,0] = DOD
# input_x1[5,0] = Type

Layer1 = load_rnn.hidden.weight.detach().cpu().numpy()
# Layer2 = load_rnn.hidden1.weight.detach().cpu().numpy()
Layer3 = load_rnn.predict.weight.detach().cpu().numpy()
#print(Layer1)
#print(Layer2)
#print(Layer3)
Bias1 = load_rnn.hidden.bias.detach().cpu().numpy().reshape(20,1)
# Bias2 = load_rnn.hidden1.bias.detach().cpu().numpy().reshape(10,1)
Bias3 = load_rnn.predict.bias.detach().cpu().numpy().reshape(1,1)
#print(load_rnn.hidden.bias)
#print(load_rnn.hidden1.bias)
#print(load_rnn.predict.bias)
First = np.matmul(Layer1,input_x1)
FirstSum = np.add(First , Bias1)
FirstSumRelu = np.maximum(FirstSum, 0)
# Second = np.matmul(Layer2,FirstSumRelu)
# SecondSum = np.add(Second , Bias2)
# SecondSumRelu = np.maximum(SecondSum, 0)
# Third = np.matmul(Layer3,SecondSumRelu)
Third = np.matmul(Layer3,FirstSumRelu)
ThirdSum = np.add(Third , Bias3)

print('Degradation value calculated by weights manually is: ',ThirdSum,"%%")

#testing 
# input_x = np.zeros((1,6))
input_x = np.zeros((1,5))
input_x[0,0] = Capacity
input_x[0,1] = Temp
input_x[0,2] = DISC
input_x[0,3] = SOC
input_x[0,4] = DOD
# input_x[0,5] = Type

# x_tensor = torch.Tensor(input_x).unsqueeze(0).cuda()  # unsqueeze gives a 1, batch_size dimension
x_tensor = torch.Tensor(input_x).unsqueeze(0)
hidden = None
test_out = load_rnn(x_tensor)          # use trained_rnn not load_rnn
Degradation = test_out.cpu().data.numpy().flatten()
print('Degradation value calculated by nn is: ',Degradation,"%%")

print('layer1', Layer1)
print('layer3', Layer3)
print('Bias1', Bias1)
print('Bias3', Bias3)

