import numpy as np
from DeepLearning.Softmax_new import Softmax
class Layer:
    def __init__(self, activation_function,weights, biases):
        self.activation_function=activation_function
        self.W_weights=weights
        self.n_NeuronsNumber,self.n_NeuronsInPrevLayer=self.W_weights.shape
        self.B_biases=biases

    def forward(self,X_samples):
        self.X=X_samples
        return self.activation_function(np.dot(self.W_weights,self.X)+self.B_biases)

class EndLayer:
    def __init__(self,activation_function,weights, biases):
        self.W_weights=weights
        self.n,self.l=self.W_weights.shape

    def forward(self,X_samples):
        self.X=X_samples
        self.m=X_samples.shape[1]
        softmax=Softmax(X_samples,self.W_weights,np.ones(self.l,self.m))
        probability=softmax.theSoftmax()