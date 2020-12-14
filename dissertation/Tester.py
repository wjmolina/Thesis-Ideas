import pickle
from Reconstructor import Reconstructor
from matplotlib import pyplot as plt
import numpy as np

images = pickle.load(open('Data/test_images.pkl', 'rb'))
bestI, bestD = None, - 1
rec = Reconstructor('Power')
for i, img in enumerate(images):
    print(i, len(images))
    result, dice, _ = rec.reconstruct(img, 0, 0)
    if dice > bestD:
        bestI, bestD, bestimg = result, dice, img
        plt.imshow(np.abs(bestI - bestimg))
        plt.show()
plt.imshow(np.abs(bestI - bestimg))
plt.show()