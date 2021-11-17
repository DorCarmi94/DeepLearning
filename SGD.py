import numpy as np
import matplotlib.pyplot as plt
from DeepLearning.Softmax_new import Softmax
class SGD:
    def __init__(self,samples,indicators,alpha,batchSize):
        n, m1 = samples.shape
        l, m2 = indicators.shape

        self.m_numberOfSamples = m2
        self.l_numberOfLayers = l
        self.n_sizeOfEachSample = n

        self.W_weights=Softmax.getWeights(self.n_sizeOfEachSample,self.l_numberOfLayers)
        self.alpha_learningRate=alpha
        self.X_samples=samples
        self.C_indicators=indicators
        self.BatchSize=batchSize
        self.samplesNumbers = list(range(0, self.m_numberOfSamples))
        np.random.shuffle(self.samplesNumbers)
        self.numberOfMiniBatches=int(self.m_numberOfSamples/self.BatchSize)


    def SGD_run(self, numberOfIters):
        totalSum=0
        wk= self.W_weights
        for epoch in range(numberOfIters):
            softmax=[]
            for j in range(0,self.numberOfMiniBatches):
                X_batch = np.array([np.array(self.X_samples[:,self.samplesNumbers[x]]) for x in range(j*self.BatchSize,(j+1)*self.BatchSize)])
                C_batch = np.array([np.array(self.C_indicators[:,self.samplesNumbers[x]]) for x in range(j*self.BatchSize,(j+1)*self.BatchSize)])
                Xt_batch=X_batch.T
                Ct_batch=C_batch.T
                softmax=Softmax(Xt_batch,Ct_batch,wk)
                grad_fi=softmax.gradientForLoss()
                grad_fi = grad_fi.T
                #sum=grad_fi*1/self.BatchSize
                grad_fi_normalized=grad_fi*1/self.BatchSize
                change=grad_fi_normalized*self.alpha_learningRate
                wk=wk-change
                print(wk)
            totalSum=totalSum+wk
        theW=totalSum/numberOfIters
        softmax_print = Softmax(self.X_samples, self.C_indicators, theW)
        print(softmax_print.loss())
        return theW


def testSGD_leastSqares():
    linear = np.linspace(1, 10, 1000).reshape(-1, 1)
    noise = np.random.normal(0, 2, linear.shape)
    newsignal = linear + noise * 2
    plt.plot(newsignal, 'bo')
    plt.plot(linear)

    C = []

    for i in range(0, len(noise)):
        if (noise[i] > linear[i]):
            C.append([0, 1])
        else:
            C.append([1, 0])

    samples=[[i,newsignal[i][0]] for i in range(len(noise))]
    samples=np.array(samples)
    sgd=SGD(np.array(samples).T,np.array(C).T,0.0000001,10)
    w=sgd.SGD_run(3)
    print(w)
    softmax_new = Softmax(samples.T,np.array(C).T, w)



if __name__ == '__main__':
    testSGD_leastSqares()


