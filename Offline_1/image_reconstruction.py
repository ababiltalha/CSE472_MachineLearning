import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approximation(A, k):
    U, S, V = np.linalg.svd(A)
    A_k = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    return A_k

image_path = "/home/ababil/BUET/4-2/CSE472/CSE472_MachineLearning/Offline_1/image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.axis('off')
# plt.show()

# print(image.shape)
# print(min(image.shape))

A = image

k_values = [1, 5, 10, 20, 25, 30, 35, 40, 50, 100, 200, min(image.shape)] # 12 values of n_components
approximations = []
for k in k_values:
    approximation = low_rank_approximation(A, k)
    approximations.append(approximation)

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(approximations[i], cmap='gray')
    ax.set_title(f'k = {k_values[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()
