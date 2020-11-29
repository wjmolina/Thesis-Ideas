import numpy as np

from scipy.special import comb as nchoosek


class Plotter:
    @staticmethod
    def get_bernstein_basis(n, a, b):
        x = np.linspace(a, b, b - a + 1)
        return [nchoosek(n, k) * ((x - a) / (b - a)) ** k * (1 - ((x - a) / (b - a))) ** (n - k) for k in range(n + 1)]

    @staticmethod
    def get_separable_basis(n, a, b, c, d):
        return [np.outer(Y, X) for X in Plotter.get_bernstein_basis(n, a, b) for Y in Plotter.get_bernstein_basis(n, c, d)]

    @staticmethod
    def construct_image(coefficients, basis):
        return (np.einsum('i, ijk -> jk', coefficients, basis) <= 0).astype(float)

    @staticmethod
    def sorensen_dice_coefficient(image, reconstruction):
        true_positive = np.sum((image == reconstruction) * image)
        false_positive = np.sum((image != reconstruction) * reconstruction)
        false_negative = np.sum((image != reconstruction) * image)
        return 2 * true_positive / (2 * true_positive + false_positive + false_negative)

    @staticmethod
    def get_nonseparable_basis(n, L):
        x, y = np.meshgrid(np.linspace(0, L, L + 1), np.linspace(0, L, L + 1))
        res = []
        for i in range(n + 1):
            for j in range(n - i + 1):
                res.append(nchoosek(n, i) * nchoosek(n - i, j) * x ** i * y ** j * (L - x - y) ** (n - i - j) / L ** n)
        return res

    @staticmethod
    def get_power_basis(n, a, b):
        X = np.linspace(a, b, b - a + 1)
        return [X ** k for k in range(n + 1)]

    @staticmethod
    def get_bivariate_basis(n, a, b, c, d):
        X, Y, B = Plotter.get_power_basis(n, a, b), Plotter.get_power_basis(n, c, d), []
        for i in range(n + 1):
            for j in range(n - i + 1):
                B.append(np.outer(Y[j], X[i]))
        return B

    @staticmethod
    def get_stably_bounded_shape(a, b, c, d, height, width):
        M = np.random.randn(3, 3)
        M = M @ M.T
        while not all(np.linalg.eig(M)[0]) > 0:
            M = np.random.randn(3, 3)
            M = M @ M.T
        x, y = np.meshgrid(np.linspace(a, b, height), - np.linspace(c, d, width))
        z = np.array([x ** i * y ** (2 - i) for i in range(3)])
        return (np.einsum('ikl,ij,jkl->kl', z, M, z) + np.einsum('i,ijk->jk', np.random.randn(10), np.array([x ** i * y ** j for i in range(4) for j in range(4 - i)])) < 0).astype(float)
