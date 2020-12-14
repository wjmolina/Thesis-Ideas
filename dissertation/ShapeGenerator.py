import pickle

import numpy as np


def is_bounded(image):
    border = set(image[:, 0]) | set(image[:, - 1]) | set(image[0, :]) | set(image[- 1, :])
    if border == {0} and 1 in image:
        return True, image
    if border == {1} and 0 in image:
        return True, 1 - image
    return False, image


def generate_images(file_name, n_images):
    x, y = np.meshgrid(np.linspace(- 2, 2, 513), np.linspace(- 2, 2, 513))
    basis = [x ** i * y ** j for i in range(5) for j in range(5 - i)]
    shapes = []
    for i in range(n_images):
        print('Generating Images', i + 1, '/', n_images)
        shape = (np.einsum('i, ijk -> jk', 2 * np.random.random(15) - 1, basis) <= 0).astype(int)
        # shape = Plotter.get_stably_bounded_shape(- 1, 1, - 1, 1, 513, 513)
        is_it_bounded, shape = is_bounded(shape)
        while not is_it_bounded:
            shape = (np.einsum('i, ijk -> jk', 2 * np.random.random(15) - 1, basis) <= 0).astype(int)
            # shape = Plotter.get_stably_bounded_shape(- 1, 1, - 1, 1, 513, 513)
            is_it_bounded, shape = is_bounded(shape)
        shapes.append(shape)
    pickle.dump(shapes, open('Data/' + file_name + '.pkl', 'wb'))
