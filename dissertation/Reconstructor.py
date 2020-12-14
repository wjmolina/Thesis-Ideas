import pickle

import numpy as np
from scipy.signal import fftconvolve

from Utilities import Plotter


class Reconstructor(Plotter):
    def __init__(self, kind):
        self.kind = kind
        if kind == 'Power':
            self.basis = self.get_bivariate_basis(4, 0, 512, 0, 512)
            try:
                print('loading data')
                self.kernel = pickle.load(open('Power/PowerKernel.pkl', 'rb'))
                self.moment = pickle.load(open('Power/PowerMomentMatrix.pkl', 'rb'))
            except:
                print('loading data failed. creating data')
                from PowerPKL import kernel, moment
                self.kernel = kernel
                self.moment = moment
        elif kind == 'Separable':
            self.basis = self.get_separable_basis(4, 0, 512, 0, 512)
            self.kernel = pickle.load(open('Separable/SeparableKernel.pkl', 'rb'))
            self.moment = pickle.load(open('Separable/SeparableMomentMatrix.pkl', 'rb'))
        elif kind == 'Non-Separable':
            self.basis = self.get_nonseparable_basis(4, 1024)
            self.kernel = pickle.load(open('NonSeparable/NonSeparableKernel.pkl', 'rb'))
            self.moment = pickle.load(open('NonSeparable/NonSeparableMomentMatrix.pkl', 'rb'))
        else:
            exit()

    def reconstruct(self, image, mean, std):
        sampling = fftconvolve(image, self.kernel)
        sampling += np.random.normal(mean, std, sampling.shape)
        if self.kind == 'Non-Separable':
            sampling = np.pad(sampling, [[0, 512]])
        M = np.einsum('ijkl, kl -> ij', self.moment, sampling)
        b = - M[:, 0]
        M = M[:, 1:]
        coefficients = np.insert(np.linalg.pinv(M) @ b, 0, 1)
        reconstruction = self.construct_image(coefficients, self.basis)
        if self.kind == 'Non-Separable':
            reconstruction = reconstruction[: 513, : 513]
            sampling = sampling[: 519, : 519]
        reconstruction = max(reconstruction, 1 - reconstruction, key=lambda x: self.sorensen_dice_coefficient(image, x))
        return reconstruction, self.sorensen_dice_coefficient(image, reconstruction), sampling[:: 16, :: 16]
