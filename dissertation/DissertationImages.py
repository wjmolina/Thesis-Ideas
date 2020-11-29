import pickle

import matplotlib.pyplot as plt

from Reconstructor import Reconstructor

shape = plt.imread('Data/art_1.png')[:, :, 0]

# plt.figure()
# plt.gray()
# plt.axis('off')
# plt.imshow(shape)
# plt.show()

# reconstructor = Reconstructor('Power')
# reconstructor = Reconstructor('Separable')
reconstructor = Reconstructor('Non-Separable')

std = .1
REC, SDC, DWN = reconstructor.reconstruct(shape, 0, std)
for i in range(100):
    rec, sdc, dwn = reconstructor.reconstruct(shape, 0, std)
    if sdc > SDC:
        REC, SDC, DWN = rec, sdc, dwn
    print(i + 1, '/', 100, SDC)

# plt.figure()
# plt.gray()
# plt.axis('off')
# plt.imshow(DWN)

plt.figure()
plt.gray()
plt.axis('off')
plt.imshow(abs(REC - shape))

plt.show()
