from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Reconstructor import Reconstructor


def is_bounded(image):
    border = set(image[:, 0]) | set(image[:, - 1]) | set(image[0, :]) | set(image[- 1, :])
    if border == {0} and 1 in image:
        return True, image
    if border == {1} and 0 in image:
        return True, 1 - image
    return False, image


xx, yy = np.meshgrid(np.linspace(- 2, 2, 513), np.linspace(- 2, 2, 513))
basis = [xx ** i * yy ** j for i in range(5) for j in range(5 - i)]
X, Y = np.meshgrid(np.linspace(- 2, 2, 32), np.linspace(- 2, 2, 32))
Basis = [X ** i * Y ** j for i in range(5) for j in range(5 - i)]
N = 20
reconstructor = [Reconstructor('Power'), Reconstructor('Separable'), Reconstructor('Non-Separable')]

while True:
    C = 2 * np.random.random(15) - 1
    shape = (np.einsum('i, ijk -> jk', C, basis) <= 0).astype(int)
    is_it_bounded, shape = is_bounded(shape)
    while not is_it_bounded:
        C = 2 * np.random.random(15) - 1
        shape = (np.einsum('i, ijk -> jk', C, basis) <= 0).astype(int)
        is_it_bounded, shape = is_bounded(shape)

    R = [0, 0, 0]
    for j in range(len(reconstructor)):
        for i in range(N):
            print(j + 1, '/', i + 1, '/', N)
            _, sdc, _ = reconstructor[j].reconstruct(shape, 0, .1)
            R[j] += sdc

    if R[0] < R[1] < R[2]:
        print(R)
        plt.imshow(shape)
        plt.show()
        break

M = 100

df = defaultdict(list)
for j in range(len(reconstructor)):
    for i in range(M):
        print(j + 1, '/', i + 1, '/', M)
        _, sdc, _ = reconstructor[j].reconstruct(shape, 0, .1)
        df['Sørensen–Dice Coefficient'].append(sdc)
        df['Method'].append('Power' if j == 0 else 'Separable' if j == 1 else 'Non-Separable')


class my_cnn(torch.nn.Module):
    def __init__(self):
        super(my_cnn, self).__init__()
        self.cv1 = torch.nn.Conv2d(1, 64, 8)
        self.fc1 = torch.nn.Linear(64 * 25 * 25, 64)
        self.fc2 = torch.nn.Linear(64, 15)
        xxx, yyy = torch.meshgrid(torch.linspace(- 3, 3, 32), torch.linspace(- 3, 3, 32))
        self.basis = torch.stack([xxx ** i * yyy ** j for i in range(5) for j in range(5 - i)])

    def forward(self, x):
        x = torch.relu(self.cv1(x))
        x = torch.relu(self.fc1(x.view(- 1, 64 * 25 * 25)))
        x = self.fc2(x)
        x = torch.sigmoid(torch.einsum('li, ijk -> ljk', x, self.basis)).view(- 1, 1, 32, 32)
        return x


Shape = (np.einsum('i, ijk -> jk', C, Basis) <= 0).astype(float)
Shape = torch.from_numpy(Shape).view(1, 1, 32, 32).float()
model = my_cnn()
model.load_state_dict(torch.load('normal_noisy_1.pth'))
for i in range(M):
    n_input = Shape
    n_output = model(Shape + torch.from_numpy(np.random.normal(0, .1, Shape.shape)).view(1, 1, 32, 32).float()).round().view(32, 32).detach().numpy()
    plt.imshow(n_input)
    plt.show()
    plt.imshow(n_output)
    plt.show()
    df['Sørensen–Dice Coefficient'].append(Reconstructor.sorensen_dice_coefficient(n_input.numpy(), n_output))
    df['Method'].append('Neural Network')
plt.show()

df = pd.DataFrame(df)
df.to_pickle('PPT.pkl')
