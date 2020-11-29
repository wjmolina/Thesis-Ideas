import pickle

import matplotlib.pyplot as plt


def shape_displayer(file_name):
    shapes = pickle.load(open('Data/' + file_name + '.pkl', 'rb'))
    for i, shape in enumerate(shapes):
        plt.figure()
        plt.gray()
        plt.axis('off')
        plt.imshow(shape)
        plt.title(i)
        plt.show()


shape_displayer('shapes_1')
