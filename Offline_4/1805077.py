import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasetPaths = ['data/2D_data_points_1.txt', 'data/2D_data_points_2.txt', 'data/3D_data_points.txt', 'data/6D_data_points.txt']

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
    V = V.T
    # take the first 2 columns of V
    V = V[:, :2]

    D = D @ V
    return D

def PCAnD(D, n):
    # Center the data
    D = D - np.mean(D, axis=0)
    U, S, V = np.linalg.svd(D, full_matrices=False)
    V = V.T
    V = V[:, :n]

    D = D @ V
    return D
    
if __name__ == "__main__":
    dataset = 3
    D = importData(dataset)
    N, M = D.shape
    if M > 2:
        D = PCAnD(D, 2)
    plt.scatter(D[:, 0], D[:, 1])
    plt.show()
    
