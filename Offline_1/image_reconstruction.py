import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approximation(A, k):
    U, S, V = np.linalg.svd(A)
    A_k = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    return A_k

# Read the image and convert it to grayscale
image = cv2.imread('image.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image to lower dimensions
resized_image = cv2.resize(gray_image, (500, 500))

# Perform Singular Value Decomposition
k_values = np.linspace(1, min(resized_image.shape), num=10, dtype=int)
approximations = []
for k in k_values:
    approximation = low_rank_approximation(resized_image, k)
    approximations.append(approximation)

# Plot the resultant k-rank approximations
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(approximations[i], cmap='gray')
    ax.set_title(f'k = {k_values[i]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
