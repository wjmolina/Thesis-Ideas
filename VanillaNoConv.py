import matplotlib.pyplot as plt
import torch

from random import choice
from scipy.special import comb as nCk

def get_stably_bounded_shapes(N, n, a, b, c, d, h, w):
    '''
    This could be made into a class for the data loader.
    '''
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
    return (I + torch.einsum('li,ijk->jk', C, B) < 0).float()

class SoloGenerator(torch.nn.Module):
    def __init__(self):
        super(SoloGenerator, self).__init__()
        self.fc1 = torch.nn.Linear(64 ** 2, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = torch.nn.Linear(4, 15)

    @staticmethod
    def construct_image(c):
        x, y = torch.meshgrid(torch.linspace(- 5, 5, 64), - torch.linspace(- 5, 5, 64))
        b = torch.stack([x ** i * y ** j for i in range(5) for j in range(5 - i)])
        return torch.einsum('ij,jkl->ikl', c, b).view(- 1, 64, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x.view(- 1, 64 ** 2)))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(SoloGenerator.construct_image(self.fc3(x)))
        return x

n = 20000

soloGenerator = SoloGenerator()

data = get_stably_bounded_shapes(1, 4, - 5, 5, - 5, 5, 64, 64)
plt.imshow(data.view(64, 64))
plt.gray()
plt.axis('off')
plt.title('image to be reconstructed')
plt.show()

optim = torch.optim.AdamW(soloGenerator.parameters())
loss_f = torch.nn.BCELoss()

stats = []

for i in range(n):
    optim.zero_grad()
    output = soloGenerator(data)
    target = data
    loss = loss_f(output, target)
    loss.backward()
    optim.step()
    if i % (n // 10) == 0:
        print(100 * i // n, '%', loss.item())
    stats.append(loss.item())

orig = data.view(64, 64)
recn = soloGenerator(data).detach().round().view(64, 64)

plt.figure()
plt.imshow(abs(orig - recn))
plt.gray()
plt.title('error')
plt.axis('off')

plt.figure()
plt.imshow(orig)
plt.gray()
plt.title('original')
plt.axis('off')

plt.figure()
plt.plot(torch.arange(len(stats)), stats)
plt.title('loss')

plt.show()