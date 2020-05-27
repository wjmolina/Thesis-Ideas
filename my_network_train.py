# This code trains my neural network.
# 05/26/20

import matplotlib.pyplot as plt
import numpy as np
import tools
import torch

from scipy import signal

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

print('generating training data')
training_size = 5000
noise_level = .01
training_x, training_y, training_n, kernel = [], [], [], tools.get_gaussian_kernel(33, 5)
for _ in range(training_size):
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

print('training')
n_epochs = 500
model = my_cnn()
model.load_state_dict(torch.load('save/noiseless_1.pth'))
optimizer = torch.optim.AdamW(model.parameters())
loss_function = torch.nn.MSELoss()
loss_data = []
for i in range(n_epochs):
    optimizer.zero_grad()
    output = model(training_y + training_n)
    target = training_x
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    print(i + 1, '/', n_epochs, loss.item())
    loss_data.append(loss.item())
torch.save(model.state_dict(), 'save/noiseless_1.pth')

print('displaying')
for i in range(10):
    n_input = training_x[i][0]
    n_output = model((training_y + training_n)[i].view(1, * (training_y + training_n)[i].shape)).round().view(64, 64).detach()
    plt.figure()
    plt.imshow(abs(n_input - n_output))
    plt.gray()
plt.figure()
plt.plot(loss_data)
plt.show()