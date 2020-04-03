import numpy as np


def get_stably_bounded_shape(a, b, c, d, height, width):
    M = np.random.randn(3, 3)
    M = M @ M.T
    while not all(np.linalg.eig(M)[0]) > 0:
        M = np.random.randn(3, 3)
        M = M @ M.T
    x, y = np.meshgrid(np.linspace(a, b, height), - np.linspace(c, d, width))
    z = np.array([x ** i * y ** (2 - i) for i in range(3)])
    return (np.einsum('ikl,ij,jkl->kl', z, M, z) + np.einsum('i,ijk->jk', np.random.randn(10), np.array([x ** i * y ** j for i in range(4) for j in range(4 - i)])) < 0).astype(float)


def get_gaussian_kernel(size, window, standard_deviation=1):
    t = np.linspace(- window, window, size)
    x, y = np.meshgrid(t, - t)
    return np.exp(- (x ** 2 + y ** 2) / (2 * standard_deviation ** 2)) / (2 * np.pi * standard_deviation ** 2) * (t[1] - t[0]) ** 2


def get_image(coefficients, basis):
    return np.einsum('i,ijk->jk', coefficients, basis)
