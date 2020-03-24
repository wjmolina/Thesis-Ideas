import matplotlib.pyplot as plt
import torch

from math import pi
from random import choice
from scipy.special import comb as nCk


def gaussian_kernel(n, v=1):
    '''
     input : positive integer, non-negative number (optional)
    output : [n, n] tensor
    '''
    T = torch.linspace(- 6, 6, n)
    X, Y = torch.meshgrid(T, - T)
    return torch.exp(- (X ** 2 + Y ** 2) / (2 * v ** 2)) / (2 * pi * v ** 2) * (T[1] - T[0]) ** 2


def convolution(image, kernel):
    '''
     input : [a, b, c] tensor, [d, e] tensor with b >= d and c >= e
    output : [a, b - d + 1, c - e + 1] tensor
    '''
    return torch.conv2d(image.view(- 1, 1, *image[0].shape), kernel.view(1, 1, *kernel.shape))[:, 0, :, :]


def get_stably_bounded_shapes(N, n, a, b, c, d, h, w):
    '''
    This could be made into a class for the data loader.

     input : positive integer, non-negative integer, number, number with a <= b, number, number with c <= d, positive integer, positive integer
    output : [N, h, w] tensor
    '''
    x, y = torch.meshgrid(torch.linspace(a, b, h), - torch.linspace(c, d, w))
    X = torch.stack([x ** (n // 2 - i) * y ** i for i in range(n // 2 + 1)])
    M = torch.tensor([])

    for _ in range(N):
        M_i = torch.randn(n // 2 + 1, n // 2 + 1)
        M_i = torch.tril(M_i) + torch.tril(M_i, diagonal=- 1).t()

        while not torch.all(torch.eig(M_i)[0][:, 0] > 0):
            M_i = torch.randn(n // 2 + 1, n // 2 + 1)
            M_i = torch.tril(M_i) + torch.tril(M_i, diagonal=- 1).t()

        M = torch.cat((M, M_i.view(1, *M_i.shape)))

    image = torch.einsum('ikl,mij,jkl->mkl', X, M, X)
    C = torch.randn(N, int(nCk(n + 2, 2)) - n - 1)
    B = torch.stack([x ** i * y ** j for i in range(n) for j in range(n - i)])
    return (image + torch.einsum('li,ijk->jk', C, B) < 0).float()


class SoloGenerator(torch.nn.Module):
    def __init__(self, in_h, in_w, ot_h, ot_w):
        super(SoloGenerator, self).__init__()
        self.fc1 = torch.nn.Linear(in_h * in_w, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 15)
        self.in_h = in_h
        self.in_w = in_w
        self.ot_h = ot_h
        self.ot_w = ot_w

    def construct_image(self, c):
        x, y = torch.meshgrid(torch.linspace(- 5, 5, self.ot_h), - torch.linspace(- 5, 5, self.ot_w))
        b = torch.stack([x ** i * y ** j for i in range(5) for j in range(5 - i)])
        return torch.einsum('ij,jkl->ikl', c, b).view(- 1, self.ot_h, self.ot_w)

    def forward(self, x):
        x = torch.relu(self.fc1(x.view(- 1, self.in_h * self.in_w)))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.construct_image(self.fc3(x)))
        return x


epochs = 20000
in_h = 64
in_w = 64
ot_h = 128
ot_w = 128

soloGenerator = SoloGenerator(in_h, in_w, ot_h, ot_w)

data = get_stably_bounded_shapes(1, 4, - 5, 5, - 5, 5, ot_h, ot_w)
plt.imshow(data[0])
plt.gray()
plt.axis('off')
plt.title('first image to be reconstructed')
plt.show()

kernel = gaussian_kernel(ot_h + 1 - in_h)
plt.imshow(kernel)
plt.gray()
plt.axis('off')
plt.title('kernel')
plt.show()

downsampled_data = convolution(data, kernel)
plt.imshow(downsampled_data[0])
plt.gray()
plt.axis('off')
plt.title('first downsample to be reconstructed')
plt.show()

noisy_downsampled_data = downsampled_data + torch.randn(downsampled_data.shape) * 0.01
# noisy_downsampled_data = (noisy_downsampled_data - noisy_downsampled_data.min()) / (noisy_downsampled_data.max() - noisy_downsampled_data.min())
plt.imshow(noisy_downsampled_data[0])
plt.gray()
plt.axis('off')
plt.title('first noisy downsample to be reconstructed')
plt.show()

optim = torch.optim.AdamW(soloGenerator.parameters())
loss_f = torch.nn.MSELoss()
# loss_f = torch.nn.BCELoss()

stats = []

for i in range(epochs):
    optim.zero_grad()
    loss = loss_f(convolution(soloGenerator(noisy_downsampled_data), kernel), noisy_downsampled_data)
    loss.backward()
    optim.step()

    if i % (epochs // 10) == 0:
        print(100 * i // epochs, '%', loss.item())

    stats.append(loss.item())

for i in range(len(data)):
    orig = data[i]
    dwns = noisy_downsampled_data[i]
    recn = soloGenerator(noisy_downsampled_data).detach().round()[i]

    plt.figure()
    plt.imshow(orig)
    plt.gray()
    plt.title('first original')
    plt.axis('off')

    plt.figure()
    plt.imshow(dwns)
    plt.gray()
    plt.title('first noisy downsample')
    plt.axis('off')

    plt.figure()
    plt.imshow(abs(orig - recn))
    plt.gray()
    plt.title('first error')
    plt.axis('off')

    plt.figure()
    plt.plot(torch.arange(len(stats)), stats)
    plt.title('loss')

    plt.show()
