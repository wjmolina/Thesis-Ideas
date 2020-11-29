import pickle

import matplotlib.pyplot as plt
from skimage.transform import rescale

from Reconstructor import Reconstructor

shapes = pickle.load(open('Data/shapes_1.pkl', 'rb'))
reconstructor = Reconstructor('Non-Separable')
N = 1000
j = 55
# A = []

for shape in shapes:
    # print(i + 1, N)
    rec, sdc, dwn = reconstructor.reconstruct(shape, 0, 0.01)
#     A.append([rec, sdc, dwn])

# A.sort(key=lambda x: x[1], reverse=True)
# A = A[:20]
# for i in range(len(A)):
#     plt.imsave('PPT/rec_{}.png'.format(i), abs(A[i][0] - shapes[j]))
#     # plt.imsave('PPT/dwn_{}.png'.format(i), rescale(A[i][2], 16, mode='edge', anti_aliasing=False, anti_aliasing_sigma=None, preserve_range=True, order=0))
