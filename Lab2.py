from functions import *

import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters # new import LAB2
from sklearn.preprocessing import Normalizer # new import LAB2
from sklearn.preprocessing import OneHotEncoder # new import LAB2

# -------------------- LAB2 - DATA PREPARATION

data = pd.read_csv('covtype.csv', sep=';')
# we use a sample with 10% of the data
dataSample = data.sample(frac=0.1)

# this gives us a table with some statistic values for each variable
register_matplotlib_converters() # a new import was needed

sb_vars = dataSample.select_dtypes(include='object')
dataSample[sb_vars.columns] = dataSample.select_dtypes(['object']).apply(lambda x: x.astype('category'))

cols_nr = dataSample.select_dtypes(include='number')
cols_sb = dataSample.select_dtypes(include='category')

print(dataSample.describe(include='all'))

# MISSING VALUES IMPUTATION
# once we don't have missing values we don't need to worry about this

# NORMALIZATION

transf = Normalizer().fit(cols_nr) # a new import was needed
cols_nr = pd.DataFrame(transf.transform(cols_nr, copy=True), columns= cols_nr.columns)
norm_data = cols_nr.join(cols_sb, how='right')
print(norm_data.describe(include='all'))

# VARIABLE DUMMIFICATION

def dummify (df, cols_to_dummify):
    one_hot_encoder = OneHotEncoder(sparse=False)

    for var in cols_to_dummify:
        one_hot_encoder.fit(data[var].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([var])
        transformed_data = one_hot_encoder.transform(data[var].values.reshape(-1, 1))
        df = pd.concat((df, pd.DataFrame(transformed_data, columns=feature_names)), 1)
        df.pop(var)
    return df


df = dummify(data, cols_sb.columns)
print(df.describe(include='all'))

# DATA BALANCING
