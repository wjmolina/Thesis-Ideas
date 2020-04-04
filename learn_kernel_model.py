import matplotlib.pyplot as plt
import numpy as np
import tools
import torch

from scipy import signal

class learn_kernel_model(torch.nn.Module):
    def __init__(self, kernel_size):
        super(learn_kernel_model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size)
    def forward(self, x):
        return torch.relu(self.conv1(x))

# Create the training dataset.
training_size = 100
training_x = []
training_y = []
training_n = []
kernel = tools.get_gaussian_kernel(33, 5)
for _ in range(training_size):
    x = tools.get_stably_bounded_shape(- 3, 3, - 3, 3, 64, 64)
    training_x.append(x)
    y = signal.fftconvolve(x, kernel, mode='valid')
    training_y.append(y)
    n = np.random.randn(* y.shape) * .0
    training_n.append(n)
training_x = np.array(training_x)
training_y = np.array(training_y)
training_n = np.array(training_n)

# Check the example.
plt.figure()
plt.title('noisy downsample')
plt.imshow(training_y[0] + training_n[0])
plt.colorbar()
plt.gray()
plt.show()

# Train and Display
n_epochs = 1000
model = learn_kernel_model(33)
optimizer = torch.optim.AdamW(model.parameters())
loss_function = torch.nn.MSELoss()
loss_history = []
for i in range(n_epochs):
    optimizer.zero_grad()
    output = model(torch.from_numpy(training_x).view(- 1, 1, * training_x[0].shape).float())
    target = torch.from_numpy(training_y + training_n).view(- 1, 1, * training_y[0].shape).float()
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    print(i, '/', n_epochs, loss.item())
output = model(torch.from_numpy(training_x[0]).view(- 1, 1, * training_x[0].shape).float()).view(32, 32).detach().numpy()
plt.figure()
plt.title('module 3 output')
plt.imshow(output)
plt.colorbar()
plt.gray()
plt.figure()
plt.title('downsample')
plt.imshow(training_y[0])
plt.colorbar()
plt.gray()
plt.figure()
plt.title('module 3 output & downsample error')
plt.imshow(abs(training_y[0] - output))
plt.colorbar()
plt.gray()
plt.figure()
plt.title('noisy downsample')
plt.imshow(training_y[0] + training_n[0])
plt.colorbar()
plt.gray()
plt.figure()
plt.title('learned kernel')
plt.imshow(torch.relu(model.conv1.weight).view(33, 33).detach())
plt.colorbar()
plt.gray()
plt.figure()
plt.title('loss')
plt.plot(loss_history[len(loss_history) // 2:])
plt.show()