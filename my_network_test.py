# This code tests my neural network.
# 05/25/20

import matplotlib.pyplot as plt
import numpy as np
import tools
import torch

from scipy import signal
from matplotlib import animation

class my_cnn(torch.nn.Module):
    def __init__(self):
        super(my_cnn, self).__init__()
        self.cv1 = torch.nn.Conv2d(1, 64, 8)
        self.fc1 = torch.nn.Linear(64 * 25 * 25, 64)
        self.fc2 = torch.nn.Linear(64, 15)
        x, y = torch.meshgrid(torch.linspace(- 3, 3, 64), torch.linspace(- 3, 3, 64))
        self.basis = torch.stack([x ** i * y ** j for i in range(5) for j in range(5 - i)])
    def forward(self, x):
        x = torch.relu(self.cv1(x))
        x = torch.relu(self.fc1(x.view(- 1, 64 * 25 * 25)))
        x = self.fc2(x)
        x = torch.sigmoid(torch.einsum('li, ijk -> ljk', x, self.basis)).view(- 1, 1, 64, 64)
        return x

print('generating data')
n_images = 5000
noise_level = 0
training_x, training_y, training_n, kernel = [], [], [], tools.get_gaussian_kernel(33, 5)
for _ in range(n_images):
    x = tools.get_stably_bounded_shape(- 3, 3, - 3, 3, 64, 64)
    training_x.append(x)
    y = signal.fftconvolve(x, kernel, mode='valid')
    training_y.append(y)
    n = np.random.randn(* y.shape) * noise_level
    training_n.append(n)
training_x = np.array(training_x)
training_x = torch.from_numpy(training_x).view(- 1, 1, * training_x[0].shape).float()
training_y = np.array(training_y)
training_y = torch.from_numpy(training_y).view(- 1, 1, * training_y[0].shape).float()
training_n = np.array(training_n)
training_n = torch.from_numpy(training_n).view(- 1, 1, * training_n[0].shape).float()

print('displaying boxplot')
model = my_cnn()
model.load_state_dict(torch.load('save/noiseless_1.pth'))
data = []
for i in range(n_images):
    n_input = training_x[i][0]
    n_output = model((training_y + training_n)[i].view(1, * (training_y + training_n)[i].shape)).round().view(64, 64).detach()
    data.append(tools.SorensenDiceCoefficient(n_input, n_output))
plt.boxplot(data)
plt.show()