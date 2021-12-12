import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import math
import random

class encoder_decoder_toyModel(nn.Module):
    def __init__(self, hidden_n,seq_size):
        super().__init__()
        self.encoder=nn.LSTM(1,hidden_n,batch_first=True)
        self.decoder=nn.LSTM(hidden_n,hidden_n,batch_first=True)
        self.linear=nn.Linear(hidden_n,1)
        self.seq_size=seq_size
        self.hidden_n=hidden_n

    def forward(self,x):
        outFromEncoder=self.encoder(x)
        _,(all_hidden,_)=outFromEncoder
        last_hidden=all_hidden.view(-1,1,self.hidden_n)
        last_h_rep=last_hidden.repeat(1,self.seq_size,1)
        Y_out,_ = self.decoder(last_h_rep)
        outFromLinear=self.linear(Y_out)
        return outFromLinear


class train_toyModel():
    def __init__(self,
                 data,validationData,
                 sequenceLength,
                 batchSize,hiddenSize,learningRate,gradientClipping,gradientClippingValue,numberOfIterations,
                 sgd_string="Adam",MSE_string="MSE"):
        self.bsz=batchSize
        self.hiddenSize=hiddenSize
        self.lr=learningRate
        self.bsz=batchSize
        self.clip=gradientClipping
        self.clipVal=gradientClippingValue
        self.iters=numberOfIterations

        self.ae=encoder_decoder_toyModel(self.hiddenSize,sequenceLength)

        if(sgd_string=="sgd"):
            self.optim=optim.SGD(self.ae.parameters(),self.lr)
        else:
            self.optim=optim.Adam(self.ae.parameters(),self.lr)

        self.x_data=data
        self.validationData=validationData
        self.MES_string=MSE_string

    def train(self):
        trainingLosses=[]
        validationLosses=[]

        GPU_or_CPU=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(GPU_or_CPU!="cpu"):
          print(f"working on GPU:{GPU_or_CPU}")
        self.ae.to(GPU_or_CPU)
        for i in range(self.iters):
            curr_loss=0
            for batch in self.x_data:
                self.optim.zero_grad()
                currBatch=batch.to(GPU_or_CPU)
                aeOut=self.ae.forward(currBatch)
                calc_andGetLoss=0
                if(self.MES_string=="MSE"):
                    calc_andGetLoss=nn.MSELoss().forward(aeOut,currBatch)
                calc_andGetLoss.backward() # calculate gradients
                self.optim.step()
                curr_loss+=calc_andGetLoss.item()
            avgLossForIter=curr_loss/len(self.x_data)
            trainingLosses.append(avgLossForIter)
            print(f"Iteration={i+1}, TrainLoss={avgLossForIter}")
        return trainingLosses,validationLosses

class LoadedData(Dataset):
  def __init__(self,tensorDataSet):
    self.data=np.expand_dims(tensorDataSet,2).astype(np.float32)
  def __getitem__(self,idx):
    return self.data[idx]
  def __len__(self):
    return len(self.data)

def load_data(path,bsz,shouldShuffle):
    tensorDataSet=pd.read_csv(path)
    dataObject=LoadedData(tensorDataSet)
    data_l=DataLoader(dataObject,batch_size=bsz,shuffle=shouldShuffle)
    return data_l


#define the parameters
p_BatchSize=64
p_HiddenStateSize=100
p_LearningRate=0.001
p_GradienClipping=False
p_GradientClippingValue=0.1
p_NumberOfIterations=10

x_data=load_data("toy_train.csv",p_BatchSize,True)
val_data=load_data("toy_val.csv",p_BatchSize,False)
test_data=load_data("toy_test.csv",p_BatchSize,False)

theTrainer=train_toyModel(x_data,val_data,50,p_BatchSize,p_HiddenStateSize,p_LearningRate,p_GradienClipping,p_GradientClippingValue,p_NumberOfIterations)
trainLosses,validationLosses=theTrainer.train()