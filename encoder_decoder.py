import numpy as np
import torch.nn as nn
import torch
from torch.utils.data.dataset import TensorDataset
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd



class encoder_decoder(nn.Module):
    def __init__(self, hidden_n,seq_size):
        super().__init__()
        self.encoder=nn.LSTM(1,hidden_n,batch_first=True)
        self.decoder=nn.LSTM(hidden_n,hidden_n,batch_first=True)
        self.linear=nn.Linear(hidden_n,1)
        self.seq_size=seq_size
        self.hidden_n=hidden_n

    def forward(self,x):
        _,(all_hidden,_)=self.encoder(x)
        last_hidden=all_hidden.view(-1,1,self.hidden_n)
        last_h_rep=last_hidden.repeat(1,self.seq_size,1)
        Y_out,_ = self.decoder(last_h_rep)
        out=self.linear(Y_out)
        return out



