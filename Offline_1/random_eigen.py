import numpy as np

n = int(input("Enter the dimension of the matrix: "))
A = np.random.randint(low=-9, high=9, size=(n, n))
while np.linalg.matrix_rank(A) != n:
    print("Rank is not full. Trying again...")
    A = np.random.randint(low=-9, high=9, size=(n, n))
print(f"Matrix A:\n{A}")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues:\n{eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

reconstructed_A = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
reconstruction_check = np.allclose(A, reconstructed_A)
print("Reconstruction check:", reconstruction_check)
