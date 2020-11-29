import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Reconstructor import Reconstructor

reconstructor = [Reconstructor('Power'), Reconstructor('Separable'), Reconstructor('Non-Separable')]
shapes = pickle.load(open('Data/shapes_1.pkl', 'rb'))
df = defaultdict(list)
for i, shape in enumerate(shapes):
    for j in range(len(reconstructor)):
        print(j + 1, i + 1, 100)
        for _ in range(50):
            for std in [10 ** (- k) for k in range(3, 8)]:
                rec, sdc = reconstructor[j].reconstruct(shape, 0, std)
                df['Sørensen–Dice Coefficient'].append(sdc)
                df['Method'].append('Power' if j == 0 else 'Separable' if j == 1 else 'Non-Separable')
                df['Standard Deviation'].append(std)
    pickle.dump(pd.DataFrame(df), open('Data/shapes_1_data.pkl', 'ab'))
    flierprops = {'marker': 'o', 'markersize': 1.25, 'markeredgewidth': 0}
    sns.boxplot(x='Standard Deviation', y='Sørensen–Dice Coefficient', hue='Method', data=df, palette='cubehelix', flierprops=flierprops)
    plt.legend(loc='lower right')
    plt.title('Shape #' + str(i))
    plt.show()
