import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from scipy import stats
from tqdm import tqdm
import numpy as np
from sklearn.mixture import GaussianMixture

class GMM():
    """
    https://blog.csdn.net/qq_52466006/article/details/127166877
    https://blog.csdn.net/qq_52466006/article/details/127186276
    """
    def __init__(self) -> None:
        self.theta = {}
        self.theta_recorder = []
        # parameter initialization
        self.theta['w1'],self.theta['w2'] = 0.5,0.5
        self.theta['mu1'],self.theta['mu2'] = 0,10
        self.theta['sigma1'],self.theta['sigma2'] = 0.8,0.8

    def E_step(self,data):
        Rz = []
        Rz1_up = self.theta['w1']*stats.norm(self.theta['mu1'],self.theta['sigma1']).pdf(data)
        Rz2_up = self.theta['w2']*stats.norm(self.theta['mu2'],self.theta['sigma2']).pdf(data)
        Rz_down = Rz1_up+Rz2_up
        Rz1 = Rz1_up/Rz_down
        Rz2 = Rz2_up/Rz_down
        Rz.append(Rz1)
        Rz.append(Rz2)
        Rz = np.array(Rz)
        return Rz

    def M_step(self,data,Rz):
        n = len(data)
        self.theta['w1'] = np.sum(Rz[0])/n
        self.theta['w2'] = np.sum(Rz[1])/n
        self.theta['mu1'] = np.sum(Rz[0]*data)/np.sum(Rz[0])
        self.theta['mu2'] = np.sum(Rz[1]*data)/np.sum(Rz[1])
        Sigma1 = np.sum(Rz[0]*np.square(data-self.theta['mu1']))/(np.sum(Rz[0]))
        Sigma2 = np.sum(Rz[1]*np.square(data-self.theta['mu2']))/(np.sum(Rz[1]))
        self.theta['sigma1'],self.theta['sigma2'] = np.sqrt(Sigma1),np.sqrt(Sigma2)

    def train(self,data,epochs):
        for i in tqdm(range(epochs)):
            Rz = self.E_step(data)
            self.M_step(data,Rz)
            self.theta_recorder.append(self.theta.copy())



if __name__ == '__main__':
    model = GMM()
    X,y = make_blobs(n_samples=1000,n_features=1,centers=[[3],[5]],cluster_std=[0.5,1],random_state=100)
    print(model.theta)
    model.train(X,100)
    print(model.theta)
