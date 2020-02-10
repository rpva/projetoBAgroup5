from functions import *

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('covtype.csv', sep=';')  # import the dataset into the dataframe, using pandas
print(data.head()) # outputs table contents into console

print(data.shape)


fig = plt.figure(figsize=(10, 7))
mv = {}
for var in data:
    mv[var] = data[var].isna().sum()
    bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var, 'nr. missing values')
fig.tight_layout()
plt.show()
