import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter
        
    def fit(self, X):
        N, M = X.shape
        self.pi = np.ones(self.n_components) / self.n_components
        self.mu = np.random.rand(self.n_components, M)
        self.sigma = np.array([np.eye(M) for _ in range(self.n_components)])
        self.gamma = np.zeros((N, self.n_components))
        
        for _ in range(self.max_iter):
            self._expectation(X)
            self._maximization(X)
            
    def _expectation(self, X):
        N, M = X.shape
        for i in range(self.n_components):
            self.gamma[:, i] = self.pi[i] * self._gaussian(X, self.mu[i], self.sigma[i])
        self.gamma /= np.sum(self.gamma, axis=1, keepdims=True)
        
    def _maximization(self, X):
        N, M = X.shape
        for i in range(self.n_components):
            Nk = np.sum(self.gamma[:, i])
            self.pi[i] = Nk / N
            self.mu[i] = np.sum(self.gamma[:, i].reshape(-1, 1) * X, axis=0) / Nk
            self.sigma[i] = np.dot((self.gamma[:, i].reshape(-1, 1) * (X - self.mu[i])).T, (X - self.mu[i])) / Nk
            
    def _gaussian(self, X, mu, sigma):
        N, M = X.shape
        X = X - mu
        return np.exp(-0.5 * np.sum(X @ np.linalg.inv(sigma) * X, axis=1)) / np.sqrt((2 * np.pi) ** M * np.linalg.det(sigma))
    
    def predict(self, X):
        return np.argmax(self._predict_proba(X), axis=1)
    
    def _predict_proba(self, X):
        N, M = X.shape
        proba = np.zeros((N, self.n_components))
        for i in range(self.n_components):
            proba[:, i] = self.pi[i] * self._gaussian(X, self.mu[i], self.sigma[i])
        return proba
    
    def score(self, X):
        return np.log(np.sum(self._predict_proba(X), axis=1))