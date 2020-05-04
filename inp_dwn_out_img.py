import matplotlib.pyplot as plt
import numpy as np
import tools
import torch

from scipy import signal
from matplotlib import animation

class my_cnn(torch.nn.Module):
    def __init__(self):
        super(my_cnn, self).__init__()
        self.cv1 = torch.nn.Conv2d(1, 32, 8)
        self.fc1 = torch.nn.Linear(32 * 25 * 25, 64)
        self.fc2 = torch.nn.Linear(64, 15)
        x, y = torch.meshgrid(torch.linspace(- 5, 5, 64), - torch.linspace(- 5, 5, 64))
        self.basis = torch.stack([x ** i * y ** j for i in range(5) for j in range(5 - i)])
    def forward(self, x):
        x = torch.relu(self.cv1(x))
        x = torch.relu(self.fc1(x.view(- 1, 32 * 25 * 25)))
        x = self.fc2(x)
        x = torch.sigmoid(torch.einsum('li,ijk->ljk', x, self.basis)).view(- 1, 1, 64, 64)
        return x

# # Create the training dataset.
# training_size = 2000
# training_x = []
# training_y = []
# training_n = []
# n_level = .01
# kernel = tools.get_gaussian_kernel(33, 5)
# for _ in range(training_size):
#     x = tools.get_stably_bounded_shape(- 3, 3, - 3, 3, 64, 64)
#     training_x.append(x)
#     y = signal.fftconvolve(x, kernel, mode='valid')
#     training_y.append(y)
#     n = np.random.randn(* y.shape) * n_level
#     training_n.append(n)
# training_x = np.array(training_x)
# training_x = torch.from_numpy(training_x).view(- 1, 1, * training_x[0].shape).float()
# training_y = np.array(training_y)
# training_y = torch.from_numpy(training_y).view(- 1, 1, * training_y[0].shape).float()
# training_n = np.array(training_n)
# training_n = torch.from_numpy(training_n).view(- 1, 1, * training_n[0].shape).float()

# # Train
# n_epochs = 1000
# model = my_cnn()
# model.load_state_dict(torch.load('model1'))
# optimizer = torch.optim.AdamW(model.parameters())
# loss_function = torch.nn.MSELoss()
# for i in range(n_epochs):
#     optimizer.zero_grad()
#     output = model(training_y + training_n)
#     target = training_x
#     loss = loss_function(output, target)
#     loss.backward()
#     optimizer.step()
#     print(i, '/', n_epochs, loss.item())
# torch.save(model.state_dict(), 'model1')

# # Display
# for i in range(10):
#     n_input = training_x[i][0]
#     n_output = model((training_y + training_n)[i].view(1, * (training_y + training_n)[i].shape)).round().view(64, 64).detach()
#     plt.figure()
#     plt.imshow(n_input)
#     plt.gray()
#     plt.figure()
#     plt.imshow(n_output)
#     plt.gray()
# plt.show()

# Test
n_level = .05
kernel = tools.get_gaussian_kernel(33, 5)
model = my_cnn()
model.load_state_dict(torch.load('model1'))
model.eval()

training_x = []
training_y = []
training_n = []

for _ in range(10):
    x = tools.get_stably_bounded_shape(- 3, 3, - 3, 3, 64, 64)
    training_x.append(x)

    y = signal.fftconvolve(x, kernel, mode='valid')
    training_y.append(y)

    n = np.random.randn(* y.shape) * n_level
    training_n.append(n)

training_x = np.array(training_x)
training_x = torch.from_numpy(training_x).view(- 1, 1, * training_x[0].shape).float()

training_y = np.array(training_y)
training_y = torch.from_numpy(training_y).view(- 1, 1, * training_y[0].shape).float()

training_n = np.array(training_n)
training_n = torch.from_numpy(training_n).view(- 1, 1, * training_n[0].shape).float()

# # plt.figure()
# # plt.imshow(network_input)
# # plt.gray()
# # plt.title('network_input')

# # plt.figure()
# # plt.imshow(original_image)
# # plt.gray()
# # plt.title('original_image')

# # plt.figure()
# # plt.imshow(abs(network_output - original_image))
# # plt.gray()
# # plt.title('err')

# # plt.show()





# fig = plt.figure(figsize=(5, 5))
# columns = 3
# rows = 5
# cnt = 0

# for i in range(1, columns * rows + 1, 3):
#     n_output = model((training_y[cnt] + training_n[cnt]).view(- 1, 1, 32, 32))
#     network_input = (training_y[cnt] + training_n[cnt]).view(32, 32)
#     network_output = n_output.view(64, 64).round().detach()
#     original_image = training_x[cnt].view(64, 64)
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(network_input)
#     plt.title('noisy downsample, snr = ' + str(round(10 * torch.log10((training_y[cnt] ** 2).mean() / (training_n[cnt] ** 2).mean()).item(), 2)))
#     plt.gray()
#     plt.axis('off')
#     plt.colorbar()
#     fig.add_subplot(rows, columns, i + 1)
#     plt.imshow(original_image)
#     plt.title('original image')
#     plt.gray()
#     plt.axis('off')
#     fig.add_subplot(rows, columns, i + 2)
#     plt.imshow(abs(network_output - original_image))
#     plt.title('reconstruction error')
#     plt.gray()
#     plt.axis('off')
#     cnt += 1

# plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
plt.axis('off')

def animate(i):
    network_input = (training_y[0] + training_n[i]).view(32, 32)
    n_output = model((training_y[0] + training_n[i]).view(- 1, 1, 32, 32))
    network_output = n_output.view(64, 64).round().detach()
    original_image = training_x[0].view(64, 64)
    fig.add_subplot(1, 2, 1)
    plt.imshow(network_input)
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(network_output)
    plt.axis('off')

ani = animation.FuncAnimation(fig, animate, np.arange(0, 9), interval=25)
ani.save('tmp.gif', writer='imagemagick', fps=15)