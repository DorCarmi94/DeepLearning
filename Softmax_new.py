import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


class Softmax:
    def __init__(self, samples, indicators,weights):
        n, m1 = samples.shape
        l, m2 = indicators.shape
        assert (m1 == m2)
        self.C_indicators = indicators  # Yt-- very confusing
        self.X_samples = samples  # Ct
        self.m_numberOfSamples = m2
        self.l_numberOfLayers = l
        self.n_sizeOfEachSample = n
        self.W_weights = weights


        self.eta = 0



    def getWeights(n_sizeOfEachSample, l_numberOfLayers):
        return np.random.default_rng().normal(0, 1, (n_sizeOfEachSample, l_numberOfLayers))
    def find_eta(self):
        maxVectorSum = 0
        for j in range(self.l_numberOfLayers):
            curr = np.dot(self.X_samples.T, self.W_weights[:, j])
            sumCurr = np.sum(curr, axis=0)
            if (sumCurr > maxVectorSum):
                maxVectorSum = sumCurr
                self.eta = np.dot(self.X_samples.T, self.W_weights[:, j])

    def loss(self):
        self.find_eta()

        another_den = 0
        for j in range(self.l_numberOfLayers):
            another_den += np.exp(np.dot(self.X_samples.T, self.W_weights[:,j])-self.eta)

        totalLoss = 0
        for k in range(self.l_numberOfLayers):
            # nominator = np.exp(np.dot(self.X_samples.T, self.W_weights[k])-np.max(np.exp(np.dot(self.W_weights, self.X_samples)), axis=0))
            nominator = np.exp(np.dot(self.X_samples.T, self.W_weights[:,k])-self.eta)
            # division=np.log(nominator / denominator)
            division = np.log(np.divide(nominator, another_den))
            totalLoss = totalLoss + np.dot(self.C_indicators[k], division)
        totalLoss = (-1) * (totalLoss / self.m_numberOfSamples)
        return totalLoss

    def gradientForLoss(self):
        self.find_eta()


        # deriviation in respect to W
        # denomintaor=np.sum(np.exp(np.dot(self.X_samples.T,self.W_weights)),axis=0)
        denomintaor = 0

        for j in range(self.l_numberOfLayers):
            denomintaor += np.exp(np.dot(self.X_samples.T, self.W_weights[:, j])-self.eta)

        collectDevForEachWp = []
        for p in range(self.l_numberOfLayers):
            nominator = np.exp(np.dot(self.X_samples.T, self.W_weights[:, p])-self.eta)
            division = nominator / denomintaor
            insideParenthesis = division - self.C_indicators[p]
            theCurrentGradForWp = np.dot(self.X_samples, insideParenthesis)
            theCurrentGradForWp = theCurrentGradForWp / self.m_numberOfSamples
            collectDevForEachWp.append(theCurrentGradForWp)
        self.LossGradient_respectTo_W = np.array(collectDevForEachWp)

        # deriviation in respect to X
        nominator = np.dot(self.W_weights.T, self.X_samples)
        denomintaor = np.sum(np.exp(np.dot(self.W_weights, self.X_samples)))

        theDerivByX = nominator / denomintaor
        theDerivByX = theDerivByX - self.C_indicators
        theDerivByX = np.dot(self.W_weights, theDerivByX)
        theDerivByX = theDerivByX / self.m_numberOfSamples
        self.LossGradient_respectTo_X = theDerivByX

        return self.LossGradient_respectTo_W

    def gradientTestForSoftMaxLoss(self):
        # m = 20
        # n=2
        n = 20
        w = self.W_weights
        d = np.random.normal(0, 1, (self.n_sizeOfEachSample, self.l_numberOfLayers))
        d = d / np.linalg.norm(d)
        print("x:")
        print(self.X_samples)
        print("------")
        print("d:")
        print(d)
        epsilon = 0.1
        F0 = self.loss()
        g0 = self.gradientForLoss()
        y0 = np.zeros(8)
        y1 = np.zeros(8)
        print("k\terror order 1 \t\t error order 2")
        for k in range(0, 8):
            # epsk = np.full((self.n_sizeOfEachSample,self.m_numberOfSamples),epsilon * (0.5 ** k))
            epsk = (epsilon * (0.5 ** k))
            self.W_weights = w + epsk * d
            Fk = self.loss()
            d_flatten_trans = (d.flatten())
            d_flatten_trans = d_flatten_trans.transpose()
            g0_flatten = g0.flatten()
            nptdot = np.dot(d_flatten_trans, g0_flatten)
            F1 = F0 + epsk * nptdot
            y0[k] = abs(Fk - F0)
            y1[k] = abs(Fk - F1)
            print(k, "\t", abs(Fk - F0), "\t", abs(Fk - F1))

        plt.figure(figsize=(8, 6))
        plt.plot(y0)
        plt.plot(y1)
        plt.legend(loc="upper right")
        plt.yscale('log')
        plt.show()
        self.W_weights = w


if __name__ == '__main__':
    dataset = loadmat("SwissRollData.mat")
    Yt = dataset["Yt"]
    Ct = dataset["Ct"]

    n, m1 = Yt.shape
    l, m2 = Ct.shape
    weights=Softmax.getWeights(n,l)
    softMax = Softmax(Yt, Ct,weights)
    softMax.gradientTestForSoftMaxLoss()
    
    
    