# Gaussian Kernel

import matplotlib.pyplot as plt
import torch

from math import pi

def gaussian_kernel(n, v=1):
    T = torch.linspace(- 3, 3, n)
    X, Y = torch.meshgrid(T, T)
    return 1 / (2 * pi * v ** 2) * torch.exp(- (X ** 2 + Y ** 2) / (2 * v ** 2))

def convolution(I, K):
    return torch.conv2d(I.view(1, 1, *I.shape), K.view(1, 1, *K.shape))[0, 0, :, :]

plt.imshow(convolution(torch.randn(64, 64), gaussian_kernel(33)))
plt.gray()
plt.axis('off')
plt.show()
