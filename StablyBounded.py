# This is clean code that generates general stably-bounded shapes.

import matplotlib.pyplot as plt
import torch

from scipy.special import comb as nCk

def get_stably_bounded_shapes(N, n, a, b, c, d, h, w):
    x, y = torch.meshgrid(torch.linspace(a, b, h), - torch.linspace(c, d, w))
    X = torch.stack([x ** (n // 2 - i) * y ** i for i in range(n // 2 + 1)])
    M = torch.tensor([])
    for _ in range(N):
        M_i = torch.randn(n // 2 + 1, n // 2 + 1)
        M_i = torch.tril(M_i) + torch.tril(M_i, diagonal=-1).t()
        while not torch.all(torch.eig(M_i)[0][:, 0] > 0):
            M_i = torch.randn(n // 2 + 1, n // 2 + 1)
            M_i = torch.tril(M_i) + torch.tril(M_i, diagonal=-1).t()
        M = torch.cat((M, M_i.view(1, n // 2 + 1, - 1)))
    I = torch.einsum('ikl,mij,jkl->mkl', X, M, X)
    C = torch.randn(N, int(nCk(n + 2, 2)) - n - 1)
    B = torch.stack([x ** i * y ** j for i in range(n) for j in range(n - i)])
    return I + torch.einsum('li,ijk->jk', C, B)

shapes = get_stably_bounded_shapes(10, 4, - 10, 10, - 10, 10, 512, 512)
for shape in shapes:
    plt.figure()
    plt.imshow((shape < 0).float())
    plt.gray()
    plt.axis('off')
plt.show()