# This code displays the self.cv1 layer.
# 05/26/20

import matplotlib.pyplot as plt
import torch

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

model = my_cnn()
model.load_state_dict(torch.load('save/noiseless_1.pth'))

f, axarr = plt.subplots(8, 8)
cnt = 0
for i in range(8):
    for j in range(8):
        axarr[i, j].imshow(model.cv1.weight[cnt].detach().reshape(8, 8))
        axarr[i, j].axis('off')
        cnt += 1
plt.show()