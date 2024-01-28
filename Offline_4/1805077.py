import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.utils.extmath import svd_flip
# from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

datasetPaths = ['data/2D_data_points_1.txt', 'data/2D_data_points_2.txt', 'data/3D_data_points.txt', 'data/6D_data_points.txt']
np.random.seed(77)

def importData(dataset):
    data = pd.read_csv(datasetPaths[dataset], sep=",", header=None)
    N, M = data.shape
    # print(N, M)
    D = data.to_numpy()
    return D

def PCA2D(D):
    # Center the data
    D = D - np.mean(D, axis=0)
    U, S, V = np.linalg.svd(D, full_matrices=False)    
    # U, V = svd_flip(U, V)
    V = V.T
    # Take the first 2 columns of V
    V = V[:, :2]

    D = np.dot(D, V)
    return D

def PCAnD(D, n):
    # Center the data
    D = D - np.mean(D, axis=0)
    U, S, V = np.linalg.svd(D, full_matrices=False)
    V = V.T
    V = V[:, :n]

    D = D @ V
    return D

def Gaussian(X, mu, sigma):
    N, M = X.shape
    X = X - mu
    probability = (1 / (np.sqrt((2 * np.pi) ** M * np.linalg.det(sigma)) + 1e-6)) * np.exp(-0.5 * np.sum(X @ np.linalg.inv(sigma) * X, axis=1))
    return probability

class GaussianMixtureModel:
    def __init__(self, numComponents, maxIter=1000, tolerance=1e-6):
        self.numComponents = numComponents
        self.maxIter = maxIter
        self.tolerance = tolerance
        
    def fit(self, D, contourAnimation):
        N, M = D.shape
        # Weights of each component
        self.w = np.ones(self.numComponents) / self.numComponents
        # Mean of each component (initialized randomly)
        self.mu = np.random.rand(self.numComponents, M)
        # Covariance matrix of each component (initialized as identity)
        self.sigma = np.array([np.eye(M) for _ in range(self.numComponents)])
        # Conditional probability of each component given the data
        self.pik = np.zeros((N, self.numComponents))
        
        self.logLikelihood = -np.inf
        for _ in range(self.maxIter):
            self.expectation(D)
            self.maximization(D)
            logLikelihood = self.calcLogLikelihood(D)
            # Contour animation
            if contourAnimation:
                plt.ion()
                plt.clf()
                plt.title(f"k = {self.numComponents}")
                labels = self.predict(D)
                plt.scatter(D[:, 0], D[:, 1], c=labels, s=1)
                for i in range(self.numComponents):
                    self.plotContour(self.mu[i], self.sigma[i], alpha=0.5)
                plt.draw()
                plt.pause(0.01)
                plt.ioff()
            
            if np.abs(self.logLikelihood - logLikelihood) < self.tolerance:
                break
            self.logLikelihood = logLikelihood
            
    def expectation(self, D):
        N, M = D.shape
        for i in range(self.numComponents):
            self.pik[:, i] = self.w[i] * Gaussian(D, self.mu[i], self.sigma[i] + np.eye(M) * 1e-6)
        self.pik /= np.sum(self.pik, axis=1, keepdims=True)
        
    def maximization(self, D):
        N, M = D.shape
        for i in range(self.numComponents):
            ni = self.pik[:, i].sum()
            self.w[i] = ni / N
            self.mu[i] = np.sum(self.pik[:, i].reshape(-1, 1) * D, axis=0) / ni
            self.sigma[i] = np.dot((self.pik[:, i].reshape(-1, 1) * (D - self.mu[i])).T, (D - self.mu[i])) / ni
            
    def calcLogLikelihood(self, D):
        N, M = D.shape
        logLikelihood = 0
        for i in range(self.numComponents):
            logLikelihood += self.w[i] * Gaussian(D, self.mu[i], self.sigma[i] + np.eye(M) * 1e-6)
        logLikelihood = np.log(logLikelihood).sum()
        return logLikelihood

    def predict(self, D):
        return np.argmax(self.predict_prob(D), axis=1)
    
    def predict_prob(self, D):
        N, M = D.shape
        pik = np.zeros((N, self.numComponents))
        for i in range(self.numComponents):
            pik[:, i] = self.w[i] * Gaussian(D, self.mu[i], self.sigma[i] + np.eye(M) * 1e-6)
        return pik
    
    # Bonus
    def plotContour(self, position, covariance, ax=None, **kwargs):
        # Draw an ellipse with a given position and covariance 
        ax = ax or plt.gca()
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, S, V = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(S)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        for nsig in range(1, 5):
            ax.add_patch(Ellipse(xy = position, width = nsig * width, height = nsig * height, angle = angle, fill = False, **kwargs))
    
    
def saveImages():
    for i in range(4):
        dataset = i
        D = importData(dataset)
        N, M = D.shape
        if M > 2:
            D = PCA2D(D)
        plt.scatter(D[:, 0], D[:, 1])
        plt.savefig(f'image/{dataset}/{dataset}_plot.png')
        plt.clf()
        
        logLikelihoods = []
        models = []
        for k in range(3, 9):
            bestLL = -np.inf
            bestGMM = None
            for _ in range(5):
                gmm = GaussianMixtureModel(numComponents=k)
                gmm.fit(D, contourAnimation=False)
                temp = gmm.logLikelihood
                if temp > bestLL:
                    bestLL = temp
                    bestGMM = gmm
            logLikelihoods.append(bestLL)
            models.append(bestGMM)
            
            labels = bestGMM.predict(D)
            plt.scatter(D[:, 0], D[:, 1], c=labels, s=1)
            plt.savefig(f'image/{dataset}/{dataset}_cluster_k={k}.png')
            plt.clf()
        
        ks = range(3, 9)
        plt.plot(ks, logLikelihoods)
        plt.xlabel("k")
        plt.ylabel("Convergence log likelihood")
        plt.savefig(f'image/{dataset}/{dataset}_LL_vs_k.png')
        plt.clf()
        
        
def main():
    dataset = 2
    D = importData(dataset)
    N, M = D.shape
    if M > 2:
        D = PCA2D(D)
    plt.scatter(D[:, 0], D[:, 1])
    plt.savefig(f'image/{dataset}_plot.png')
    
    
    k=7
    
    gmm = GaussianMixtureModel(numComponents=k)
    gmm.fit(D, contourAnimation=True)
    labels = gmm.predict(D)
    plt.scatter(D[:, 0], D[:, 1], c=labels, s=1)
    plt.savefig(f'image/{dataset}_cluster_k={k}.png')
    plt.show()
    
    
if __name__ == "__main__":
    # saveImages()
    main()
