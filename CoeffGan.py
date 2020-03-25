import matplotlib.pyplot as plt
import torch

class deg_4_algebraic_shapes(torch.utils.data.Dataset):
    '''
    This needs to be cleaned.
    '''
    def __init__(self, a, b, h, w, noise_length, size):
        self.shapes = torch.FloatTensor([])
        basis = deg_4_algebraic_shapes.get_basis(a, b, h, w)
        count = 0
        while count != size:
            shape = (deg_4_algebraic_shapes.construct_image(deg_4_algebraic_shapes.get_stably_bounded_coeff(), basis) < 0).float()
            if deg_4_algebraic_shapes.is_bounded(shape)[0]:
                self.shapes = torch.cat((self.shapes, 2 * shape - 1))
                count += 1
        self.noise = torch.rand(size, noise_length)

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, index):
        return self.shapes[index], self.noise[index]

    @staticmethod
    def get_stably_bounded_coeff():
        coeff = 2 * torch.rand(16) - 1
        return torch.FloatTensor([coeff[0], coeff[1], coeff[2], coeff[3], coeff[13] ** 2 + coeff[14] ** 2 + coeff[15] ** 2, coeff[4], coeff[5], coeff[6], 2 * coeff[11] * coeff[13] + 2 * coeff[12] * coeff[14], coeff[7], coeff[8], 2 * coeff[10] * coeff[13] + coeff[12] ** 2 + coeff[11] ** 2, coeff[9], 2 * coeff[10] * coeff[11], coeff[10] ** 2]).view(1, - 1)

    @staticmethod
    def is_bounded(shapes):
        result = torch.FloatTensor([])    
        for algebraicShape in shapes:
            border = set(algebraicShape[:, 0].tolist()) | set(algebraicShape[:, - 1].tolist()) | set(algebraicShape[0, :].tolist()) | set(algebraicShape[- 1, :].tolist())
            result = torch.cat((result, torch.FloatTensor([[len(border) == 1 and 0 in border and torch.max(algebraicShape) == 1]]).float()))
        return result

    @staticmethod
    def get_basis(a, b, h, w):
        x, y = torch.meshgrid(- torch.linspace(a, b, h), torch.linspace(a, b, w))
        return torch.stack([x ** i * y ** j for i in range(5) for j in range(5 - i)])

    @staticmethod
    def construct_image(coeffs, basis):
        return torch.einsum('ij,jkl->ikl', coeffs, basis)

class D(torch.nn.Module):
    def __init__(self, d_in):
        super(D, self).__init__()
        self.d_in = d_in * d_in
        self.fc1 = torch.nn.Linear(self.d_in, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x.view(- 1, self.d_in)))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class G(torch.nn.Module):
    def __init__(self, g_in, g_out):
        super(G, self).__init__()
        self.g_in = g_in
        self.g_out = g_out
        self.fc1 = torch.nn.Linear(self.g_in, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 15)

    def alg_img(self, coeffs):
        # bdd_coeffs = torch.stack([torch.FloatTensor([coeff[0], coeff[1], coeff[2], coeff[3], coeff[13] ** 2 + coeff[14] ** 2 + coeff[15] ** 2, coeff[4], coeff[5], coeff[6], 2 * coeff[11] * coeff[13] + 2 * coeff[12] * coeff[14], coeff[7], coeff[8], 2 * coeff[10] * coeff[13] + coeff[12] ** 2 + coeff[11] ** 2, coeff[9], 2 * coeff[10] * coeff[11], coeff[10] ** 2]) for coeff in coeffs])
        x, y = torch.meshgrid(torch.linspace(- 2, 2, self.g_out), - torch.linspace(- 2, 2, self.g_out))
        basis = torch.stack([x ** i * y ** j for i in range(5) for j in range(5 - i)])
        return torch.einsum('ij,jkl->ikl', coeffs, basis).view(- 1, self.g_out ** 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x.view(- 1, self.g_in)))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.alg_img(self.fc3(x)))

d_in = 64
d = D(d_in)
d_optim = torch.optim.SGD(d.parameters(), lr=0.001, momentum=0.9)

g_in = 25
g_out = 64
g = G(g_in, g_out)
g_optim = torch.optim.SGD(g.parameters(), lr=0.001, momentum=0.9)

loss_fn = torch.nn.BCELoss()

epochs = 500
d_epochs = 5
g_epochs = 5

training_size = 1000
training_batch_size = 100
training_set = deg_4_algebraic_shapes(- 2, 2, d_in, d_in, g_in, training_size)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=training_batch_size, shuffle=True)

real_target = lambda x: torch.ones(len(x), 1)
fake_target = lambda x: torch.zeros(len(x), 1)

# Training

plt.imshow(training_set.shapes[0])
plt.gray()
plt.title('original')
plt.show()

for epoch in range(epochs):
    d_loss_r, d_loss_cnt = 0, 0

    for _ in range(d_epochs):
        for real_batch, noise_batch in training_generator:
            d_optim.zero_grad()

            d_loss = loss_fn(d(real_batch), real_target(real_batch))
            d_loss.backward()

            d_loss = loss_fn(d(g(noise_batch).detach()), fake_target(noise_batch))
            d_loss.backward()

            d_optim.step()
            d_loss_r += d_loss.item()
            d_loss_cnt += 1

    g_loss_r, g_loss_cnt = 0, 0

    for _ in range(g_epochs):
        for _, noise_batch in training_generator:
            g_optim.zero_grad()

            g_loss = loss_fn(d(g(noise_batch)), real_target(noise_batch))
            g_loss.backward()
            
            g_optim.step()
            g_loss_r += g_loss.item()
            g_loss_cnt += 1

    print('d', '%f' % (d_loss_r / d_loss_cnt), 'g', '%f' % (g_loss_r / g_loss_cnt), epoch + 1, '/', epochs)
    # torch.save(d.state_dict(), 'GanSanityCheckD.pt')
    # torch.save(g.state_dict(), 'GanSanityCheckG.pt')

# Testing

plt.figure()
plt.imshow(training_set.shapes[0])
plt.gray()
plt.title('original')

plt.figure()
plt.imshow(g(training_set.noise[0]).detach().view(64, 64))
plt.gray()
plt.title('reconstruction')

plt.show()
