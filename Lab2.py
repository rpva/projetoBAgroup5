from functions import *

import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters # new import LAB2
from sklearn.preprocessing import Normalizer # new import LAB2
from sklearn.preprocessing import OneHotEncoder # new import LAB2
from imblearn.over_sampling import SMOTE, RandomOverSampler # new import LAB2

# -------------------- LAB2 - DATA PREPARATION

data = pd.read_csv('covtype.csv', sep=';')
# we use a sample with 10% of the data
dataSample = data.sample(frac=0.1)

# (22) this gives us a table with some statistic values for each variable
register_matplotlib_converters() # a new import was needed

sb_vars = dataSample.select_dtypes(include='object')
dataSample[sb_vars.columns] = dataSample.select_dtypes(['object']).apply(lambda x: x.astype('category'))

cols_nr = dataSample.select_dtypes(include='number')
cols_sb = dataSample.select_dtypes(include='category')

print(dataSample.describe(include='all'))

# MISSING VALUES IMPUTATION
# (23 and 24) once we don't have missing values we don't need to worry about this

# NORMALIZATION

# (25)
transf = Normalizer().fit(cols_nr) # a new import was needed
cols_nr = pd.DataFrame(transf.transform(cols_nr, copy=True), columns= cols_nr.columns)
norm_data = cols_nr.join(cols_sb, how='right')
print(norm_data.describe(include='all'))

# VARIABLE DUMMIFICATION it shouldn't be applied to the class variable, since it will transform a simple multi label
# classification problem into a multiclass problem

# (26)
def dummify (df, cols_to_dummify): # REMOVE VARIABLE CLASS
    one_hot_encoder = OneHotEncoder(sparse=False)

    for var in cols_to_dummify:
        one_hot_encoder.fit(dataSample[var].values.reshape(-1, 1))
        feature_names = one_hot_encoder.get_feature_names([var])
        transformed_data = one_hot_encoder.transform(dataSample[var].values.reshape(-1, 1))
        df = pd.concat((df, pd.DataFrame(transformed_data, columns=feature_names)), 1)
        df.pop(var)
    return df


df = dummify(dataSample, cols_sb.columns)
print(df.describe(include='all'))

# DATA BALANCING

# (27)
# unbal = pd.read_csv('data/unbalanced.csv', sep=';') # this is an unbalanced dataset
unbal = dataSample
target_count = unbal['Cover_Type(Class)'].value_counts()
plt.figure()
plt.title('Cover_Type(Class)')
plt.bar(target_count.index, target_count.values)
plt.show()

# min_class = target_count.idxmin()
# ind_min_class = target_count.index.get_loc(min_class)

print('Minority class:', target_count[4])
print('Majority class:', target_count[2])
print('Proportion:', round(target_count[4] / target_count[2], 2), ': 1')

# (28)
ind_min_class = 4
ind_max_class = 2
min_class = target_count[ind_min_class]
max_class = target_count[ind_max_class]

values = {'Original': [target_count.values[ind_min_class], target_count.values[2]]}

df_class_min = unbal[unbal['Cover_Type(Class)'] == min_class]
df_class_max = unbal[unbal['Cover_Type(Class)'] == max_class]

df_under = df_class_max.sample(len(df_class_min))
values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

df_over = df_class_min.sample(len(df_class_max), replace=True)
values['OverSample'] = [len(df_over), target_count.values[ind_max_class]]

smote = SMOTE(random_state=42)
y = unbal.pop('Cover_Type(Class)').values
X = unbal.values
_, smote_y = smote.fit_sample(X, y)
smote_target_count = pd.Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[ind_max_class]]

plt.figure()
multiple_bar_chart(plt.gca(),
                        [target_count.index[ind_min_class], target_count.index[ind_max_class]], values, 'Target', 'Frequency', 'Cover_Type(Class)')
plt.show()
