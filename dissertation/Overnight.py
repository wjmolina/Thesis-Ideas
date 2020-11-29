import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from DataGenerator import generate_data
from ShapeGenerator import generate_images

for i in range(1, 5):
    file_name = 'shapes_' + str(i)
    generate_images(file_name)
    generate_data(file_name)

for i in range(1, 501):
    sns.boxplot(x='Standard Deviation', y='Sørensen–Dice Coefficient', hue='Method', data=pickle.load(open('Data/shapes_' + str(i) + '_data.pkl', 'rb')))
    plt.show()
