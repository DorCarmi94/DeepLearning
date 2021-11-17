import numpy as np
import matplotlib.pyplot as plt
from DeepLearning.Softmax_new import Softmax
from sklearn.model_selection import train_test_split
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
                #print(wk)
            totalSum=totalSum+wk
        theW=totalSum/numberOfIters
        softmax_print = Softmax(self.X_samples, self.C_indicators, theW)
        print(softmax_print.loss())
        return theW


def testSGD_leastSqares():
    linear = np.linspace(start=1, stop=10, num=100).reshape(-1, 1)
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

    samples=[[i/len(noise),newsignal[i][0]] for i in range(len(noise))]
    samples=np.array(samples)
    X=np.array(samples).T
    Y=np.array(C).T
    X_train,X_test,y_train, y_test = train_test_split(X,Y)
    sgd=SGD(X_train,y_train,0.001,100)
    newW=sgd.SGD_run(3)
    for i in range(1000):
        sgd.W_weights=newW
        newW=sgd.SGD_run(3)

    print("predictions:\n")
    print(np.dot(X_test.T,newW).T)

    print("\n\nActual:")
    print(y_test)

def plot_graph_accuracy_vs_epoch(x_axis_t, y_axis_t, x_axis_v, y_axis_v, batch_size, eta, num_of_epochs):
    plt.plot(x_axis_t, y_axis_t, color='green', label='Training Set', linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='green', markersize=0)
    plt.plot(x_axis_v, y_axis_v, color='blue', label='Validation Set',  linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='blue', markersize=0)
    # for limiting the x axis from 1
    x_axis_1 = np.zeros(2)
    y_axis_1 = np.zeros(2)
    y_axis_1[1] = 1
    plt.plot(x_axis_1, y_axis_1, color='blue', linestyle='solid', linewidth=0,
             marker='o', markerfacecolor='blue', markersize=0)
    plt.xlim(1, num_of_epochs)
    m = mt.PercentFormatter(1)
    plt.gca().yaxis.set_major_formatter(m)
    plt.xlabel("Number of Epochs")
    plt.ylabel('Accuracy Percentage')
    plt.legend()
    plt.title('Learning rate= %.2f' % eta + '       Accuracy VS Epochs        Batch size =%d' % batch_size)
    plt.show()

if __name__ == '__main__':
    testSGD_leastSqares()


