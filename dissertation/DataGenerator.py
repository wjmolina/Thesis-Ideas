import pickle
from collections import defaultdict

import pandas as pd

from Reconstructor import Reconstructor


def generate_data(file_name):
    reconstructor = [Reconstructor('Power'), Reconstructor('Separable'), Reconstructor('Non-Separable')]
    shapes = pickle.load(open('Data/' + file_name + '.pkl', 'rb'))
    df = defaultdict(list)
    for j in range(len(reconstructor)):
        for i, shape in enumerate(shapes):
            print(j + 1, '/', i + 1, '/', len(shapes))
            for std in [.1 for _ in range(5)]:
                _, sdc, _ = reconstructor[j].reconstruct(shape, 0, std)
                df['Sørensen–Dice Coefficient'].append(sdc)
                df['Method'].append('Power' if j == 0 else 'Separable' if j == 1 else 'Non-Separable')
                df['Standard Deviation'].append(std)
    df = pd.DataFrame(df)
    df.to_pickle('Data/' + file_name + '_data.pkl')


generate_data('shapes_1')
