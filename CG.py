import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import minimize
from scipy.special import expit

def get_stably_bounded_image(a, b, c, d, h, w):
    M = np.random.randn(3, 3)
    M = M @ M.T
    while not all(np.linalg.eig(M)[0]) > 0:
        M = np.random.randn(3, 3)
        M = M @ M.T
    x, y = np.meshgrid(np.linspace(a, b, h), - np.linspace(c, d, w))
    z = np.array([x ** i * y ** (2 - i) for i in range(3)])
    return (np.einsum('ikl,ij,jkl->kl', z, M, z) + np.einsum('i,ijk->jk', np.random.randn(10), np.array([x ** i * y ** j for i in range(4) for j in range(4 - i)])) < 0).astype(float)    

def get_gaussian_kernel(hw, sd=1):
    t = np.linspace(- 10, 10, hw)
    x, y = np.meshgrid(t, - t)
    return np.exp(- (x ** 2 + y ** 2) / (2 * sd ** 2)) / (2 * np.pi * sd ** 2) * (t[1] - t[0]) ** 2

def construct_image(coeffs, basis):
    return expit(np.einsum('i,ijk->jk', coeffs, basis))

def reconstruct(ndwspl, ker, orig, basis):
    loss = lambda coeffs: np.square(np.subtract(signal.fftconvolve(construct_image(coeffs, basis), ker, mode='valid'), ndwspl)).mean()
    result = minimize(loss, np.random.randn(15) * 3, method='CG', options={'disp': True})
    return result.x

orig = get_stably_bounded_image(- 5, 5, - 5, 5, 512, 512)

plt.imshow(orig)
plt.gray()
plt.axis('off')
plt.title('orig')
plt.show()

ker = get_gaussian_kernel(257)
dwspl = signal.fftconvolve(orig, ker, mode='valid')
ndwspl = dwspl + np.random.randn(* dwspl.shape) * .05

plt.imshow(ndwspl)
plt.gray()
plt.axis('off')
plt.title('ndwspl')
plt.show()

x, y = np.meshgrid(np.linspace(- 5, 5, 512), - np.linspace(- 5, 5, 512))
basis = np.array([x ** i * y ** j for i in range(5) for j in range(5 - i)])

coeffs = reconstruct(ndwspl, ker, orig, basis)

plt.imshow(abs(construct_image(coeffs, basis).round() - orig))
plt.gray()
plt.axis('off')
plt.title('err')
plt.show()