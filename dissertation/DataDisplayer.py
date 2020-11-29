import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_pickle('PPT.pkl')
# df = df.sort_values('Sørensen–Dice Coefficient', ascending=False).groupby(['Method', 'Standard Deviation']).head(500)
flierprops = {'marker': 'o', 'markersize': 1.25, 'markeredgewidth': 0}
sns.boxplot(x='Method', y='Sørensen–Dice Coefficient', data=df, palette='cubehelix', flierprops=flierprops)
plt.legend(loc='lower right')
plt.show()
