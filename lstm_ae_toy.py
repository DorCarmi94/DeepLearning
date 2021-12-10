import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from encoder_decoder_toyModel import encoder_decoder_toyModel
from sklearn.model_selection import train_test_split


def create_syntetic_data(dim_in=1000,dim_out=50):
    matrix=np.random.randn(dim_in,dim_out)
    matrix=np.array([m / np.linalg.norm(m) for m in matrix])

    sqweeze =np.concatenate([matrix[:,:20],0.1*matrix[:,20:30],matrix[:,30:]],axis=1)
    #
    # plt.plot(sqweeze[0], label='First Order')
    # plt.show()
    return sqweeze

def upload_to_file():
    sqweezed_data=create_syntetic_data()
    file=open("test_file.csv",'w')
    writer = csv.writer(file)

    # write a row to the csv file
    writer.writerow(sqweezed_data)
    file.close()

def load_data(bs):
    # sqweezed_data=create_syntetic_data()
    tensorDataSet=pd.read_csv("toy_train.csv")
    tensorDataSet=np.expand_dims(tensorDataSet,2).astype(np.float32)
    data_l=DataLoader(tensorDataSet,batch_size=bs)
    return data_l

def upload():
    rng = np.random.default_rng()
    data = rng.random(size=(10_000, 50))
    data = data - data.mean(axis=1, keepdims=True) + 0.5
    df = pd.DataFrame(data, columns=[f"x_{i:02}" for i in range(50)])
    X_train, X_test = train_test_split(df, test_size=0.4)
    X_val, X_test = train_test_split(X_test, test_size=0.5)
    X_train.to_csv("toy_train.csv", index=False)
    X_val.to_csv("toy_val.csv", index=False)
    X_test.to_csv("toy_test.csv", index=False)

# upload()


class train_toyModel():
    def __init__(self,sgd_string,MSE_string,data,validationData,numberOfIterations,learningRate):
        self.ae=encoder_decoder_toyModel(250,50)
        self.learningRate=learningRate
        if(sgd_string=="sgd"):
            self.optim=optim.SGD(self.ae.parameters(),self.learningRate)
        else:
            self.optim=optim.Adam(self.ae.parameters(),self.learningRate)

        self.x_data=data # tensor
        self.validationData=validationData
        self.numOfIters=numberOfIterations
        self.MES_string=MSE_string

    def train(self):
        trainingLosses=[]
        validationLosses=[]
        sampledBatch=[]
        sampledBatchOutput=[]
        GPU_or_CPU=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ae.to(GPU_or_CPU)
        for i in range(self.numOfIters):
            loss=0
            countBatches=0
            for batch in self.x_data:
                self.optim.zero_grad()
                currBatch=batch.to(GPU_or_CPU)
                aeOut=self.ae.forward(currBatch)
                calc_andGetLoss=0
                if(self.MES_string=="MSE"):
                    calc_andGetLoss=nn.MSELoss().forward(aeOut,currBatch)
                # else:
                    #todo: check about NLL loss
                    #calc_andGetLoss=nn.NLLLoss().forward(nn.LogSoftmax(1)(aeOut),currBatch)
                    # m=nn.LogSoftmax(dim=2)
                    # loss=nn.NLLLoss()
                    # calc_andGetLoss=loss(m(aeOut),currBatch)
                calc_andGetLoss.backward() # calculate gradients
                self.optim.step()
                loss+=calc_andGetLoss.item()

                if(countBatches==10):
                    sampledBatch=currBatch
                    sampledBatchOutput=aeOut
                countBatches+=1
            avgLossForIter=loss/len(self.x_data)
            trainingLosses.append(avgLossForIter)
        return trainingLosses,sampledBatch,sampledBatchOutput

def startToRun():
    batchSize=50
    data=load_data(batchSize)
    train=train_toyModel("sgd","MSE",data,data,4,0.001)
    training_losses,sampledBatch,sampledBatchOutput=train.train()

    plt.figure()
    plt.title("Loss")
    plt.plot(range(len(training_losses)),training_losses)
    plt.show()

    plt.figure()
    plt.title("signals")
    a=sampledBatch.detach()
    b=a.cpu()
    c=b.squeeze()

    aa = sampledBatchOutput.detach()
    bb = aa.cpu()
    cc = bb.squeeze()

    for bt1,bt2 in zip(c,cc):
        plt.plot(bt1)
        plt.plot(bt2)
        break
    plt.show()


    # plt.plot(cc)
    # plt.show()

if __name__ == '__main__':
    startToRun()
# encoder.forward(data_l)
# for b in data_l:
#     encoder.forward(b)
# # encoder.forward(np.expand_dims(data_l.tensors[0],2))