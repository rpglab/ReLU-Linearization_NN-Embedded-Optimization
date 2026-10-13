# -*- coding: utf-8 -*-
"""
P-CTAR-based neural network-embedded microgrid day-ahead scheduling model.

This script implements the penalized convex triangle area relaxation
(P-CTAR) of ReLU activations for the neural network-based battery degradation
model.

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
import numpy as np
import pandas as pd
import csv
import os
import time
import torch
from torch import nn

start = time.time()

pd.set_option('display.max_columns', None)

L_ceff = 1
r0 = 1

model = AbstractModel()
    
### set
model.PERIOD = Set()
model.GEN = Set()

### Generater Parameters
model.gen_min = Param(model.GEN)
model.gen_max = Param(model.GEN)
model.gen_RRlimit = Param(model.GEN)
model.gen_Kom = Param(model.GEN)
model.gen_NlCost = Param(model.GEN)
model.gen_SuCost = Param(model.GEN)

model.Layer1_Neuron_Number = RangeSet(20)
model.Layer2_Neuron_Number = RangeSet(10)
model.Layer3_Neuron_Number = RangeSet(1)

# Neural network parameters
model.Layer1_Relu_Indicator = Var(model.PERIOD, model.Layer1_Neuron_Number, within = Binary)
model.Layer2_Relu_Indicator = Var(model.PERIOD, model.Layer2_Neuron_Number, within = Binary)
model.Lowerbound =Param(default = -1000)
model.Upperbound =Param(default = 1000)

#model.Layer1_weights = Param(model.Layer1_Neuron_Number, model.Layer1_Input) # Matrix of Nueron number by Input sequence
model.Layer1_weights1 = Param(model.Layer1_Neuron_Number) # Matrix of Nueron number by Input sequence
model.Layer1_weights2 = Param(model.Layer1_Neuron_Number)
model.Layer1_weights3 = Param(model.Layer1_Neuron_Number)
model.Layer1_weights4 = Param(model.Layer1_Neuron_Number)
model.Layer1_weights5 = Param(model.Layer1_Neuron_Number)
model.Layer1_weights6 = Param(model.Layer1_Neuron_Number)

#model.Layer2_weights = Param(model.Layer2_Neuron_Number, model.Layer1_Neuron_Number) # Matrix of Nueron number by Input sequence
model.Layer2_weights1 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights2 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights3 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights4 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights5 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights6 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights7 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights8 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights9 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights10 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights11 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights12 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights13 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights14 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights15 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights16 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights17 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights18 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights19 = Param(model.Layer2_Neuron_Number)
model.Layer2_weights20 = Param(model.Layer2_Neuron_Number)

#model.Layer3_weights = Param(model.Layer3_Neuron_Number, model.Layer2_Neuron_Number) # Matrix of Nueron number by Input sequence
model.Layer3_weights1 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights2 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights3 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights4 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights5 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights6 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights7 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights8 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights9 = Param(model.Layer3_Neuron_Number)
model.Layer3_weights10 = Param(model.Layer3_Neuron_Number)

model.UBLay1 = Param(model.Layer1_Neuron_Number)
model.LBLay1 = Param(model.Layer1_Neuron_Number)

model.UBLay2 = Param(model.Layer2_Neuron_Number)
model.LBLay2 = Param(model.Layer2_Neuron_Number)

model.Layer1_bias = Param(model.Layer1_Neuron_Number) # Matrix of Each Nueron's bias
model.Layer2_bias = Param(model.Layer2_Neuron_Number) # Matrix of Each Nueron's bias 
model.Layer3_bias = Param(model.Layer3_Neuron_Number) # Matrix of Each Nueron's bias 

model.Layer1_Neuron_Value = Var(model.PERIOD, model.Layer1_Neuron_Number, domain = Reals) # 24hour by Nueron number
model.Layer2_Neuron_Value = Var( model.PERIOD, model.Layer2_Neuron_Number, domain = Reals) # 24hour by Nueron number
model.Layer3_Neuron_Value = Var(model.PERIOD, model.Layer3_Neuron_Number, domain = Reals) # 24hour by Nueron number

model.Layer1_Neuron_Activate_Value = Var(model.PERIOD, model.Layer1_Neuron_Number, domain = Reals) # 24hour by Nueron number
model.Layer2_Neuron_Activate_Value = Var(model.PERIOD, model.Layer2_Neuron_Number, domain = Reals) # 24hour by Nueron number


model.Ppv = Param(default=300)
model.Pwt = Param(default=5)
model.Esslimit_Max = Param(default=300)
model.Esslimit_Min = Param(default=0)
model.Gridlimit = Param(default=1500)
model.DeltaE = Param(default=30)
model.DeltaPgrid = Param(default=50)
model.DeltaPess = Param(default=0)
model.Pess_Max = Param(default=150)#*pow((1-ReducePercentage),(K-1)))
model.Pess_Min = Param(default=20)
model.ESSinitial = Param(default=200)
#model.ESSinitial = Var()
model.R = Param(default = 0.03)
model.degra =Param(default = 0.02)
# model.Cost_SOC =Param(default=20)
model.ESSE = Param(default=0.9)
model.ESSFeedBackLimit = Param(default = 933)#*pow((1-ReducePercentage),(K-1)))
model.ESSFeedBackLimit1 = Param(default = 722.2)#*pow((1-ReducePercentage),(K-1)))

### cost parameters
model.PgridCost = Param(model.PERIOD)
model.SelltoGrid = Param(default = 0.8)

### period parameters
model.Time_TotalPd = Param(model.PERIOD)
model.PV_Ava = Param(model.PERIOD)
model.WT_Ava = Param(model.PERIOD)
model.Penalty = Param(default = 10)

### Variables
model.u = Var(model.GEN,model.PERIOD, within =Binary)
model.v = Var(model.GEN,model.PERIOD, within =Binary)
model.USCHARGE = Var(model.PERIOD, within =Binary)
model.USDISCHARGE = Var(model.PERIOD, within =Binary)
model.UBUYFGRID = Var(model.PERIOD, within =Binary)
model.USELLTGRID = Var(model.PERIOD, within =Binary)
model.Vg = Var(model.PERIOD, within =Binary)
model.Pg = Var(model.GEN, model.PERIOD)
model.Grid = Var(model.PERIOD)
model.BUYFGRID = Var(model.PERIOD)
model.SELLTGRID = Var(model.PERIOD)
model.PCHARGE = Var(model.PERIOD)
model.PDISCHARGE = Var(model.PERIOD)
model.ESS = Var(model.PERIOD)
model.PPV = Var(model.PERIOD)
model.PWT = Var(model.PERIOD)
model.SOC = Var(model.PERIOD)
model.Temp = Param(model.PERIOD)
model.DOD = Var(model.PERIOD)
model.Rate = Var(model.PERIOD)
model.LinearD = Var(domain = Reals)
model.Capacity = Param(default=0.9)
model.Type = Param(default=1)
model.PenaltyCost = Var(domain = Reals)
model.Vcharge = Var(model.PERIOD, within =Binary)
model.Vdischarge = Var(model.PERIOD, within =Binary)

###objective function
def objfunction(model):
    totalcost = sum(model.gen_Kom[i]*(model.Pg[i,j] - model.gen_min[i])+ model.gen_NlCost[i]*model.u[i,j]+ model.gen_SuCost[i]*model.v[i,j] for i in model.GEN for j in model.PERIOD) + sum(model.BUYFGRID[j]*model.PgridCost[j]-model.SELLTGRID[j]*model.PgridCost[j]*model.SelltoGrid for j in model.PERIOD) \
        + sum(model.Layer3_Neuron_Value[i,j]*model.Esslimit_Max*200*(model.USDISCHARGE[i]+model.USCHARGE[i]) for i in model.PERIOD for j in model.Layer3_Neuron_Number)/10000*2 + sum(model.Layer1_Neuron_Activate_Value[i,j]for i in model.PERIOD for j in model.Layer1_Neuron_Number)*model.Penalty \
                +sum(model.Layer2_Neuron_Activate_Value[i,j]for i in model.PERIOD for j in model.Layer2_Neuron_Number)*model.Penalty
    return totalcost    
model.Cost = Objective(rule = objfunction, sense = minimize)

### Constrains
#power balance
def Degradation_Sum(model):
    return model.LinearD == sum(model.Layer3_Neuron_Value[i,j]*(model.USDISCHARGE[i]+model.USCHARGE[i])for i in model.PERIOD for j in model.Layer3_Neuron_Number)/100
model.Cons_degradation_sum = Constraint(rule = Degradation_Sum)

def Penalty_Cost(model):
    return model.PenaltyCost == sum(model.Layer1_Neuron_Activate_Value[i,j]for i in model.PERIOD for j in model.Layer1_Neuron_Number)*model.Penalty \
                +sum(model.Layer2_Neuron_Activate_Value[i,j]for i in model.PERIOD for j in model.Layer2_Neuron_Number)*model.Penalty
model.Cons_Penalty_Cost = Constraint(rule = Penalty_Cost)

def powerbalance(model,j):
    power = sum(model.Pg[i,j] for i in model.GEN) + model.BUYFGRID[j]+ model.PDISCHARGE[j] + model.Ppv*model.PV_Ava[j] + model.Pwt*model.WT_Ava[j]
    power = power - (model.Time_TotalPd[j] + model.SELLTGRID[j] + model.PCHARGE[j])
    return  power == 0
model.Cons_powerblance = Constraint(model.PERIOD, rule = powerbalance)

def ESSbalance(model,j):
    if j >=2:    
        expr = model.ESS[j] -model.ESS[j-1] + model.PDISCHARGE[j]/model.ESSE-model.PCHARGE[j]*model.ESSE
        return expr ==0
    else:
        return Constraint.Skip
model.Cons_ESSBalance = Constraint(model.PERIOD, rule = ESSbalance)

def ESSinitial(model, j):
    expr = model.ESS[1] -model.ESSinitial + model.PDISCHARGE[1]-model.PCHARGE[1]
    return expr == 0
model.Cons_ESSinitial = Constraint(model.PERIOD, rule = ESSinitial)

def SOC(model,j):
    expr = model.ESS[j]/model.Esslimit_Max - model.SOC[j]
    return expr == 0
model.Cons_SOC = Constraint(model.PERIOD, rule = SOC)



def Ratelimit(model,i):
    expr = (model.PDISCHARGE[i] + model.PCHARGE[i])/model.Esslimit_Max/2 - model.Rate[i]
    return expr == 0
model.Cons_Ratelimit = Constraint(model.PERIOD, rule = Ratelimit)

def DODlimit(model,i):
    expr = (model.PDISCHARGE[i] + model.PCHARGE[i])/model.Esslimit_Max - model.DOD[i]
    return expr == 0
model.Cons_DODlimit = Constraint(model.PERIOD, rule = DODlimit)
    
# def ESSReduceUsage(model,j):
#     expr = sum(model.PCHARGE[j] for j in model.PERIOD)+ sum(model.PDISCHARGE[j] for j in model.PERIOD)
#     return expr <= model.ESSFeedBackLimit
# model.Cons_ESSReduceUsage = Constraint(model.PERIOD, rule = ESSReduceUsage)
   
# def ESSReduceUsage2(model,j):
#     expr = model.PCHARGE[3] + model.PCHARGE[4] + model.PCHARGE[5] + model.PDISCHARGE[18] + model.PDISCHARGE[21]
#     return expr <= model.ESSFeedBackLimit1
# model.Cons_ESSReduceUsage2 = Constraint(model.PERIOD, rule = ESSReduceUsage2)

def ESSkeepon(model,j):
    expr =  model.USCHARGE[j] + model.USDISCHARGE[j]
    return expr <= 1
model.Cons_ESSKeepon = Constraint(model.PERIOD, rule = ESSkeepon)

def Essend(model, j):
    expr = model.ESS[24]
    return expr == model.ESSinitial
model.Cons_Essend = Constraint(model.PERIOD, rule = Essend)

'''
### limit cycle
def Vcharge1(model,j):
    if j >=2:
        expr = model.USCHARGE[j] + model.USCHARGE[j-1]
        return  expr >= model.Vcharge[j]
    else:
        return Constraint.Skip
model.Cons_Vcharge1 = Constraint(model.PERIOD,rule = Vcharge1)

def Vcharge2(model,j):
    if j >=2:
        expr = model.USCHARGE[j] - model.USCHARGE[j-1]
        return  expr <= model.Vcharge[j]
    else:
        return Constraint.Skip
model.Cons_Vcharge2 = Constraint(model.PERIOD,rule = Vcharge2)

def Vcharge3(model,j):
    if j >=2:
        expr = model.USCHARGE[j-1] - model.USCHARGE[j]
        return  expr <= model.Vcharge[j]
    else:
        return Constraint.Skip
model.Cons_Vcharge3 = Constraint(model.PERIOD,rule = Vcharge3)

def Vcharge4(model,j):
    if j >=2:
        expr = 2 - model.USCHARGE[j-1] - model.USCHARGE[j]
        return  expr >= model.Vcharge[j]
    else:
        return Constraint.Skip
model.Cons_Vcharge4 = Constraint(model.PERIOD,rule = Vcharge4)

def Vcharge5(model,j):
    return model.Vcharge[1] == 0
model.Cons_Vcharge5 = Constraint(model.PERIOD,rule = Vcharge5)

def Vdischarge1(model,j):
    if j >=2:
        expr = model.USDISCHARGE[j] + model.USDISCHARGE[j-1]
        return  expr >= model.Vdischarge[j]
    else:
        return Constraint.Skip
model.Cons_Vdischarge1 = Constraint(model.PERIOD,rule = Vdischarge1)

def Vdischarge2(model,j):
    if j >=2:
        expr = model.USDISCHARGE[j] - model.USDISCHARGE[j-1]
        return  expr <= model.Vdischarge[j]
    else:
        return Constraint.Skip
model.Cons_Vdischarge2 = Constraint(model.PERIOD,rule = Vdischarge2)

def Vdischarge3(model,j):
    if j >=2:
        expr = model.USDISCHARGE[j-1] - model.USDISCHARGE[j]
        return  expr <= model.Vdischarge[j]
    else:
        return Constraint.Skip
model.Cons_Vdischarge3 = Constraint(model.PERIOD,rule = Vdischarge3)

def Vdischarge4(model,j):
    if j >=2:
        expr = 2 - model.USDISCHARGE[j-1] - model.USDISCHARGE[j]
        return  expr >= model.Vdischarge[j]
    else:
        return Constraint.Skip
model.Cons_Vdischarge4 = Constraint(model.PERIOD,rule = Vdischarge4)

def Vdisharge5(model,j):
    return model.Vdischarge[1] == 0
model.Cons_Vdischarge5 = Constraint(model.PERIOD,rule = Vdisharge5)

def CycleLimit(model):
    expr = sum(model.Vcharge[t]+model.Vdischarge[t] for t in model.PERIOD)
    return expr <= 5
model.Cons_cyclelimit = Constraint(rule = CycleLimit)
'''

def backup(model, j):
    return model.R*model.Time_TotalPd[j] <= model.Gridlimit -model.BUYFGRID[j] + model.SELLTGRID[j] + sum(model.u[i,j]*model.gen_max[i]-model.Pg[i,j] for i in model.GEN for j in model.PERIOD)
model.Cons_backup =Constraint(model.PERIOD,rule = backup)

def genlimit1(model,i, j):
    return model.u[i,j]*model.gen_min[i] <= model.Pg[i,j]
model.Cons_genMax1 = Constraint(model.GEN,model.PERIOD,rule = genlimit1)

def genlimit2(model,i, j):
    return  model.Pg[i,j]<= model.u[i,j]*model.gen_max[i]
model.Cons_genMax2 = Constraint(model.GEN,model.PERIOD,rule = genlimit2)

def genramp1(model,i,j):
    if j >=2:
        expr = model.Pg[i,j] - model.Pg[i,j-1] - model.gen_RRlimit[i]
        return  expr <=0
    else:
        return Constraint.Skip
model.Cons_genramp1 = Constraint(model.GEN,model.PERIOD,rule = genramp1)

def genramp2(model,i,j):
    if j >=2:
        expr = model.Pg[i,j-1] - model.Pg[i,j] - model.gen_RRlimit[i]
        return   expr <=0
    else:
        return Constraint.Skip
model.Cons_genramp2 = Constraint(model.GEN,model.PERIOD,rule = genramp2)


def Esslimit1(model,j):
    return  model.ESS[j] <= model.Esslimit_Max
model.Cons_ESSlimit1 = Constraint(model.PERIOD,rule = Esslimit1)

def Esslimit2(model,j):
    return model.Esslimit_Min <= model.ESS[j]
model.Cons_ESSlimit2 = Constraint(model.PERIOD,rule = Esslimit2)

def PCHARGElimit1(model,j):
    return model.USCHARGE[j]*(model.Pess_Min) <= model.PCHARGE[j] 
model.Cons_PCHARGElimit1 = Constraint(model.PERIOD,rule = PCHARGElimit1)

def PCHARGElimit2(model,j):
    return  model.PCHARGE[j] <= model.USCHARGE[j]*(model.Pess_Max-model.DeltaPess)
model.Cons_PCHARGElimit2 = Constraint(model.PERIOD,rule = PCHARGElimit2)

def PDISCHARGElimit1(model,j):
    return model.USDISCHARGE[j]*(model.Pess_Min) <= model.PDISCHARGE[j] 
model.Cons_PDISCHARGElimit1 = Constraint(model.PERIOD,rule = PDISCHARGElimit1)

def PDISCHARGElimit2(model,j):
    return model.PDISCHARGE[j] <= model.USDISCHARGE[j]*(model.Pess_Max-model.DeltaPess)
model.Cons_PDISCHARGElimit2 = Constraint(model.PERIOD,rule = PDISCHARGElimit2)

def BGridlimit1(model,j):
    return 0 <= model.BUYFGRID[j]
model.Cons_BGridlimit1 = Constraint(model.PERIOD,rule = BGridlimit1)

def BGridlimit2(model,j):
    return model.BUYFGRID[j] <= model.UBUYFGRID[j]*(model.Gridlimit - model.DeltaPgrid)
model.Cons_BGridlimit2 = Constraint(model.PERIOD,rule = BGridlimit2)

def SGridlimit1(model,j):
    return 0 <= model.SELLTGRID[j]
model.Cons_SGridlimit1 = Constraint(model.PERIOD,rule = SGridlimit1)

def SGridlimit2(model,j):
    return model.SELLTGRID[j] <= model.USELLTGRID[j]*(model.Gridlimit - model.DeltaPgrid)
model.Cons_SGridlimit2 = Constraint(model.PERIOD,rule = SGridlimit2)

def GridStatus(model,j):
    return model.UBUYFGRID[j] + model.USELLTGRID[j] <= 1
model.Cons_GridStatus = Constraint(model.PERIOD,rule = GridStatus)

def GridExchange(model,j):
    return model.Grid[j] == model.BUYFGRID[j]-model.SELLTGRID[j]
model.Cons_GridExchange = Constraint(model.PERIOD,rule =GridExchange)
 

def genUV1(model,i,j):
    if j >=2:
        expr = model.u[i,j] - model.u[i,j-1] - model.v[i,j]
        return expr <= 0
    else:
        return Constraint.Skip
model.Cons_genUV1 = Constraint(model.GEN,model.PERIOD,rule = genUV1)

def genUV2(model,i,j):
    if j >=2:
        expr = model.u[i,j-1] - model.u[i,j] - model.v[i,j]
        return expr <= 0
    else:
        return Constraint.Skip
model.Cons_genUV2 = Constraint(model.GEN,model.PERIOD,rule = genUV2)

def geninitial(model,i):
    expr = model.u[i,1] - model.v[i,1]
    return expr <= 0
model.Cons_geninitial = Constraint(model.GEN, rule = geninitial)


# model.PERIOD represents 24 hours in the following constriants
def Layer1Calculation(model,i,j):
    expr = model.Capacity*model.Layer1_weights1[j]+model.Temp[i]/32*model.Layer1_weights2[j] \
        + model.Rate[i]*model.Layer1_weights3[j] + model.SOC[i]*model.Layer1_weights4[j] \
            + model.DOD[i]*model.Layer1_weights5[j] + model.Type*model.Layer1_weights6[j] \
                + model.Layer1_bias[j] - model.Layer1_Neuron_Value[i,j]
    return expr == 0
model.Cons_Layer1Calculation = Constraint(model.PERIOD, model.Layer1_Neuron_Number, rule = Layer1Calculation)

def Layer1ActivationA(model,i,j):
    return  model.Layer1_Neuron_Activate_Value[i,j] <= model.Layer1_Neuron_Value[i,j]*model.UBLay1[j]/(model.UBLay1[j]-model.LBLay1[j]) - model.UBLay1[j]*model.LBLay1[j]/(model.UBLay1[j]-model.LBLay1[j])
model.Cons_Layer1ActivationA = Constraint(model.PERIOD, model.Layer1_Neuron_Number, rule = Layer1ActivationA)

def Layer1ActivationB(model,i,j):
    return  model.Layer1_Neuron_Value[i,j] <= model.Layer1_Neuron_Activate_Value[i,j]
model.Cons_Layer1ActivationB = Constraint(model.PERIOD, model.Layer1_Neuron_Number, rule = Layer1ActivationB)

# def Layer1ActivationC(model,i,j):
#     return  model.Layer1_Neuron_Activate_Value[i,j] <= model.Upperbound*model.Layer1_Relu_Indicator[i,j]
# model.Cons_Layer1ActivationC = Constraint(model.PERIOD, model.Layer1_Neuron_Number, rule = Layer1ActivationC)

def Layer1ActivationD(model,i,j):
    return  0 <= model.Layer1_Neuron_Activate_Value[i,j]
model.Cons_Layer1ActivationD = Constraint(model.PERIOD, model.Layer1_Neuron_Number, rule = Layer1ActivationD)

#def Layer2Calculation(model,i,k):
#    expr = sum(model.Layer1_Neuron_Activate_Value[i,j]*model.Layer2_weights[k,j]for j in model.Layer1_Neuron_Number) \
#        + model.Layer2_bias[k] - model.Layer2_Neuron_Value[i,k]
#    return expr == 0
#model.Cons_Layer2Calculation = Constraint(model.PERIOD, model.Layer2_Neuron_Number, rule = Layer2Calculation)

def Layer2Calculation(model,i,k):
    expr = model.Layer1_Neuron_Activate_Value[i,1]*model.Layer2_weights1[k] \
        + model.Layer1_Neuron_Activate_Value[i,2]*model.Layer2_weights2[k] \
            + model.Layer1_Neuron_Activate_Value[i,3]*model.Layer2_weights3[k] \
                + model.Layer1_Neuron_Activate_Value[i,4]*model.Layer2_weights4[k] \
                    + model.Layer1_Neuron_Activate_Value[i,5]*model.Layer2_weights5[k] \
                        + model.Layer1_Neuron_Activate_Value[i,6]*model.Layer2_weights6[k] \
                            + model.Layer1_Neuron_Activate_Value[i,7]*model.Layer2_weights7[k] \
                                + model.Layer1_Neuron_Activate_Value[i,8]*model.Layer2_weights8[k] \
                                    + model.Layer1_Neuron_Activate_Value[i,9]*model.Layer2_weights9[k] \
                                        + model.Layer1_Neuron_Activate_Value[i,10]*model.Layer2_weights10[k] \
                                            + model.Layer1_Neuron_Activate_Value[i,11]*model.Layer2_weights11[k] \
                                                + model.Layer1_Neuron_Activate_Value[i,12]*model.Layer2_weights12[k] \
                                                    + model.Layer1_Neuron_Activate_Value[i,13]*model.Layer2_weights13[k] \
                                                        + model.Layer1_Neuron_Activate_Value[i,14]*model.Layer2_weights14[k] \
                                                            + model.Layer1_Neuron_Activate_Value[i,15]*model.Layer2_weights15[k] \
                                                                + model.Layer1_Neuron_Activate_Value[i,16]*model.Layer2_weights16[k] \
                                                                    + model.Layer1_Neuron_Activate_Value[i,17]*model.Layer2_weights17[k] \
                                                                        + model.Layer1_Neuron_Activate_Value[i,18]*model.Layer2_weights18[k] \
                                                                            + model.Layer1_Neuron_Activate_Value[i,19]*model.Layer2_weights19[k] \
                                                                                + model.Layer1_Neuron_Activate_Value[i,20]*model.Layer2_weights20[k] + model.Layer2_bias[k] \
                                                                                    - model.Layer2_Neuron_Value[i,k]
    return expr == 0
model.Cons_Layer2Calculation = Constraint(model.PERIOD, model.Layer2_Neuron_Number, rule = Layer2Calculation)

def Layer2ActivationA(model,i,k):
    return  model.Layer2_Neuron_Activate_Value[i,k] <= model.Layer2_Neuron_Value[i,k]*model.UBLay2[k]/(model.UBLay2[k]-model.LBLay2[k]) - model.UBLay2[k]*model.LBLay2[k]/(model.UBLay2[k]-model.LBLay2[k])
model.Cons_Layer2ActivationA = Constraint(model.PERIOD, model.Layer2_Neuron_Number, rule = Layer2ActivationA)

def Layer2ActivationB(model,i,k):
    return  model.Layer2_Neuron_Value[i,k] <= model.Layer2_Neuron_Activate_Value[i,k]
model.Cons_Layer2ActivationB = Constraint(model.PERIOD, model.Layer2_Neuron_Number, rule = Layer2ActivationB)

# def Layer2ActivationC(model,i,k):
#     return  model.Layer2_Neuron_Activate_Value[i,k] <= model.Upperbound*model.Layer2_Relu_Indicator[i,k]
# model.Cons_Layer2ActivationC = Constraint(model.PERIOD, model.Layer2_Neuron_Number, rule = Layer2ActivationC)

def Layer2ActivationD(model,i,k):
    return  0 <= model.Layer2_Neuron_Activate_Value[i,k]
model.Cons_Layer2ActivationD = Constraint(model.PERIOD, model.Layer2_Neuron_Number, rule = Layer2ActivationD)  

#def Layer3Calculation(model,i,m):
#   expr = sum(model.Layer2_Neuron_Activate_Value[i,k]*model.Layer3_weights[k,m]for k in model.Layer2_Neuron_Number) \
#       - model.Layer3_Neuron_Value[i,m]
#    return expr == 0
#model.Cons_Layer3Calculation = Constraint(model.PERIOD, model.Layer3_Neuron_Number, rule = Layer2Calculation)

def Layer3Calculation(model,i,k):
    expr = model.Layer2_Neuron_Activate_Value[i,1]*model.Layer3_weights1[k] \
        + model.Layer2_Neuron_Activate_Value[i,2]*model.Layer3_weights2[k] \
            + model.Layer2_Neuron_Activate_Value[i,3]*model.Layer3_weights3[k] \
                + model.Layer2_Neuron_Activate_Value[i,4]*model.Layer3_weights4[k] \
                    + model.Layer2_Neuron_Activate_Value[i,5]*model.Layer3_weights5[k] \
                        + model.Layer2_Neuron_Activate_Value[i,6]*model.Layer3_weights6[k] \
                            + model.Layer2_Neuron_Activate_Value[i,7]*model.Layer3_weights7[k] \
                                + model.Layer2_Neuron_Activate_Value[i,8]*model.Layer3_weights8[k] \
                                    + model.Layer2_Neuron_Activate_Value[i,9]*model.Layer3_weights9[k] \
                                        + model.Layer2_Neuron_Activate_Value[i,10]*model.Layer3_weights10[k] + model.Layer3_bias[k] \
                                            - model.Layer3_Neuron_Value[i,k]
    return expr == 0
model.Cons_Layer3Calculation = Constraint(model.PERIOD, model.Layer3_Neuron_Number, rule = Layer3Calculation)





SCUC_instance = model.create_instance('case17.dat')
SCUCsolver = SolverFactory('gurobi')
SCUCsolver.options['TimeLimit'] = 600
SCUCsolver.options.mipgap = 0.00
results = SCUCsolver.solve(SCUC_instance)
end = time.time()
Data = []
genunit = []

# for j in SCUC_instance.PERIOD:
#     X = [str(SCUC_instance.BUYFGRID[j]()), str(SCUC_instance.SELLTGRID[j]()), str(SCUC_instance.PDISCHARGE[j]()-SCUC_instance.PCHARGE[j]()),str(SCUC_instance.ESS[j]()),str(SCUC_instance.SOC[j]()),int(SCUC_instance.USDISCHARGE[j]()),int(SCUC_instance.USCHARGE[j]()),str(SCUC_instance.Layer3_Neuron_Value[j,1]())]
#     Data.append(X)
#     #Y = [str(SCUC_instance.u[1,j]()),str(SCUC_instance.v[1,j]()),str(SCUC_instance.Pg[1,j]()),str(SCUC_instance.u[2,j]()),str(SCUC_instance.v[2,j]()),str(SCUC_instance.Pg[2,j]()),str(SCUC_instance.u[3,j]()),str(SCUC_instance.v[3,j]()),str(SCUC_instance.Pg[3,j]())]
#     Y = [str(SCUC_instance.Pg[1,j]()),str(SCUC_instance.Pg[2,j]()),str(SCUC_instance.Pg[3,j]())]
#     genunit.append(Y)
# Data = pd.DataFrame(Data, columns=['BUYFGRID', 'SELLTGRID', 'BESS','ESS','SOC','UD','UC','Degradation'])
# genunit =pd.DataFrame(genunit,columns=['pg1','pg2','pg3'])
# # Z = sum(float(Data['CHARGE']))
# # print (genunit)
# print (Data)
print('Time is {}'.format(end - start))
PG = []
genunit = []
ESS_NN =[]
DOD=[]
re = []
PBESS = []
ESS_SOC =[]
print("\nresults.Solution.Status: " + str(results.Solution.Status))
print("results.solver.status: " + str(results.solver.status))
print("results.solver.termination_condition: " + str(results.solver.termination_condition))
print("results.solver.termination_message: " + str(results.solver.termination_message))
print('minimize cost: ' + str(SCUC_instance.Cost()))

load_rnn = torch.load('trained_nn_1d.pt', map_location=torch.device('cpu'))
TotalDegradation = []
DegradationCost = []
OperationalCost = []
RemainingCapacity = []
Degradation =[]
SingleD = []
Capacity = 0.9
Type = 1
for j in range (1,2):
    input_x = np.zeros((1,6))
    input_x[0,0] = Capacity
    input_x[0,1] = SCUC_instance.Temp[j]/32
    input_x[0,2] = (float(SCUC_instance.PCHARGE[j].value)+float(SCUC_instance.PDISCHARGE[j].value))/300/2
    input_x[0,3] = 0.6667
    input_x[0,4] = abs(SCUC_instance.SOC[j].value-0.666666)
    input_x[0,5] = Type   
    x_tensor = torch.Tensor(input_x).unsqueeze(0).to('cpu')
    hidden = None
    #test_out, test_h = load_rnn(x_tensor, hidden)
    test_out = load_rnn(x_tensor)
    Degradation = test_out.cpu().data.numpy().flatten()
    #print(Degradation)


for j in range (2,25):
    #SingleDegradation = BatteryDegradation(Capacity, SCUC_instance.Temp[j], (Data.iat(j-1,2)+Data.iat(j-1,3))/500, SCUC_instance.SOC[j], SCUC_instance.SOC[j-1], Type)
    input_x = np.zeros((1,6))
    input_x[0,0] = Capacity
    input_x[0,1] = SCUC_instance.Temp[j]/32
    input_x[0,2] = (float(SCUC_instance.PCHARGE[j].value)+float(SCUC_instance.PDISCHARGE[j].value))/300/2
    input_x[0,3] = SCUC_instance.SOC[j-1].value
    input_x[0,4] = abs(SCUC_instance.SOC[j].value-SCUC_instance.SOC[j-1].value)
    input_x[0,5] = Type   
    x_tensor = torch.Tensor(input_x).unsqueeze(0).to('cpu')
    hidden = None
    #test_out, test_h = load_rnn(x_tensor, hidden)
    test_out = load_rnn(x_tensor)
    SingleD = test_out.cpu().data.numpy().flatten()
    #TotalDegradation = Degradation
    Degradation = np.append(Degradation, SingleD)
    #print(SingleD)
TotalDegradation = np.append(TotalDegradation, sum(Degradation)/100)
RemainingCapacity = np.append(RemainingCapacity,90-sum(Degradation)/100)
DegradationCost = np.append(DegradationCost, sum(Degradation)*300*200/10000*2)
OperationalCost = np.append(OperationalCost, SCUC_instance.Cost())






print('\nobjective cost=', SCUC_instance.Cost())
print('Penalty cost: ', SCUC_instance.PenaltyCost())
print('total cost=', SCUC_instance.Cost()-SCUC_instance.PenaltyCost())
print('real cost=', SCUC_instance.Cost()-SCUC_instance.PenaltyCost()+DegradationCost-SCUC_instance.LinearD()*300*200/10000*2*100)
# print('real degradation cost=', DegradationCost)
print('Degradation by NN linearization method: ' + str(SCUC_instance.LinearD())+ "%")
print('Degradation retreived from NN : ' + str(TotalDegradation) + "%")
# print('Degradation error: ' , (TotalDegradation-SCUC_instance.LinearD())/SCUC_instance.LinearD())

# print('Degradation of each time=', Degradation)
