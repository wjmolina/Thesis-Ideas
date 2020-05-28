# This code trains my neural network.
# 05/26/20

import matplotlib.pyplot as plt
import numpy as np
import tools
import torch

from scipy import signal

def SorensenDiceCoefficient(I, R):
    TP = sum(sum((I == R) * I)).item()
    FP = sum(sum((I != R) * R)).item()
    FN = sum(sum((I != R) * I)).item()
    return 1 if 2 * TP + FP + FN == 0 else 2 * TP / (2 * TP + FP + FN)

class my_small_cnn(torch.nn.Module):
    def __init__(self, k):
        super(my_small_cnn, self).__init__()
        self.cv1 = torch.nn.Conv2d(1, k, 8)
        self.fc1 = torch.nn.Linear(k * 25 * 25, 15)
        self.k = k
        x, y = torch.meshgrid(torch.linspace(- 3, 3, 64), torch.linspace(- 3, 3, 64))
        self.basis = torch.stack([x ** i * y ** j for i in range(5) for j in range(5 - i)])
    def forward(self, x):
        x = torch.relu(self.cv1(x))
        x = self.fc1(x.view(- 1, self.k * 25 * 25))
        x = torch.sigmoid(torch.einsum('li, ijk -> ljk', x, self.basis)).view(- 1, 1, 64, 64)
        return x

print('generating data')
training_size = 5000
noise_level = 0
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
n_epochs = 1000
k = 32
model = my_small_cnn(k)
try:
    model.load_state_dict(torch.load('save/noiseless_small_' + str(k) + '.pth'))
    print('model loaded')
except:
    print('model not loaded')
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
torch.save(model.state_dict(), 'save/noiseless_small_' + str(k) + '.pth')
plt.figure()
plt.plot(loss_data)
plt.show()