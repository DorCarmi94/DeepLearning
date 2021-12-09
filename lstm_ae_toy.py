import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.data.dataset import TensorDataset
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from encoder_decoder import encoder_decoder
from sklearn.model_selection import train_test_split
import pandas as pd

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

def load_data():
    # sqweezed_data=create_syntetic_data()
    tensorDataSet=pd.read_csv("toy_train.csv")
    tensorDataSet=np.expand_dims(tensorDataSet,2).astype(np.float32)
    data_l=DataLoader(tensorDataSet,batch_size=1)
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

data_l=load_data()
encoder=encoder_decoder(250,50)
# encoder.forward(data_l)
for b in data_l:
    encoder.forward(b)
# # encoder.forward(np.expand_dims(data_l.tensors[0],2))