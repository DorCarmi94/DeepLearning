import numpy as np
from DeepLearning.Softmax_new import Softmax
class SGD:
    def __init__(self,samples,indicators,alpha,batchSize):
        self.W_weights=Softmax.getWeights()
        self.alpha_learningRate=alpha
        self.X_samples=samples
        self.S_miniBatch
        self.C_indicators=indicators
        self.BatchSize=batchSize
        self.m_numberOfSamples=samples.shape[0]
        self.samplesNumbers = list(range(0, self.m_numberOfSamples))
        np.random.shuffle(self.samplesNumbers)
        self.numberOfMiniBatches=self.m_numberOfSamples/self.BatchSize


    def SGD_run(self):
        wk=0
        for j in range(self.numberOfMiniBatches):
            X_batch = [self.X_samples[:,self.samplesNumbers[x]] for x in range(j*self.BatchSize,(j+1)*self.BatchSize)]
            C_batch = [self.C_indicators[:,self.samplesNumbers[x]] for x in range(j*self.BatchSize,(j+1)*self.BatchSize)]
            softmax=Softmax(X_batch,C_batch,self.W_weights)
            grad_fi=softmax.gradientForLoss()
            sum=grad_fi*1/self.BatchSize
            sum=sum*self.alpha_learningRate
            



