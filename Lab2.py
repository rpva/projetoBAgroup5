from functions import *
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder
from imblearn.over_sampling import SMOTE

import pandas as pd
import matplotlib.pyplot as plt


# -------------------- LAB2 - DATA PREPARATION

# import the dataset into the dataframe, using pandas
data = pd.read_csv('covtype.csv', sep=';')
np.random.seed(2)

dataSample = data.sample(frac=0.1)
# (22) this gives us a table with some statistic values for each variable
register_matplotlib_converters()

sb_vars = dataSample.select_dtypes(include='object')  # choose object variables from DataFrame and convert into symbolic
dataSample[sb_vars.columns] = dataSample.select_dtypes(['object']).apply(lambda x: x.astype('category'))

cols_nr = dataSample.select_dtypes(include='number')  # choose the numeric variables
cols_sb = dataSample.select_dtypes(include='category')  # choose categorical variables (there aren't any in the sample)

print(dataSample.describe(include='all'))  # describe all the variables


# MISSING VALUES IMPUTATION
# (23 and 24) since there aren't any missing values, there is no need to worry about their imputation


# NORMALIZATION
# (25) convert into normalized Gauss distribution N(0,1)
# transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(cols_nr)
# cols_nr = pd.DataFrame(transf.transform(cols_nr, copy=True), columns=cols_nr.columns)
# norm_data_zscore = cols_nr.join(cols_sb, how='right')
# norm_data_zscore.describe(include='all')

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(cols_nr)
df_nr = pd.DataFrame(transf.transform(cols_nr), columns=cols_nr.columns)
norm_data_minmax = df_nr.join(cols_sb, how='right')
norm_data_minmax.describe(include='all')

fig, axs = plt.subplots(1, 2, figsize=(20, 10), squeeze=False)
axs[0, 0].set_title('Original data')
df_nr.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 1])
plt.show()


# VARIABLE DUMMIFICATION it shouldn't be applied to the class variable, since it will transform a simple multi label
# classification problem into a multiclass problem

# (26) Dummy variables are only applied to categorical variables, but the data used consists only of numeric variables
# def dummify(df, cols_to_dummify):  # REMOVE CLASS VARIABLE
#     one_hot_encoder = OneHotEncoder(sparse=False)
#     for var in cols_to_dummify:
#         one_hot_encoder.fit(dataSample[var].values.reshape(-1, 1))
#         feature_names = one_hot_encoder.get_feature_names([var])
#         transformed_data = one_hot_encoder.transform(dataSample[var].values.reshape(-1, 1))
#         df = pd.concat((df, pd.DataFrame(transformed_data, columns=feature_names)), 1)
#         df.pop(var)
#     return df
#
#
# df = dummify(dataSample, cols_sb.columns)
# print(df.describe(include='all'))


# DATA BALANCING
# (27) unbalanced dataset: the classes are not equiprobable. It requires balancing in order to proceed with the work
dataSample = data.sample(frac=0.1)
unbalanced = dataSample
target_count = unbalanced['Cover_Type(Class)'].value_counts()
plt.figure()
plt.title('Cover_Type(Class) balance')
plt.bar(target_count.index, target_count.values)
plt.show()

min_class = target_count.idxmin()
max_class = target_count.idxmax()

print('Minority class:', target_count[min_class])
print('Majority class:', target_count[max_class])
print('Proportion:', round(target_count[min_class] / target_count[max_class], 2), ': 1')

ind_min_class = target_count.index.get_loc(min_class)
ind_max_class = target_count.index.get_loc(max_class)


# (28) split the dataset into subdatasets, one for each class of the class variable (Cover Type, in this case)
values = {'Original': [target_count.values[ind_min_class], target_count.values[ind_max_class]]}

df_class_min = unbalanced[unbalanced['Cover_Type(Class)'] == min_class]  # DataFrame
df_class_max = unbalanced[unbalanced['Cover_Type(Class)'] == max_class]
# passar de min e max para 7 classes

df_under = df_class_max.sample(len(df_class_min))
values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

df_over = df_class_min.sample(len(df_class_max), replace=True)
values['OverSample'] = [len(df_over), target_count.values[ind_max_class]]

smote = SMOTE(random_state=42)
y = unbalanced.pop('Cover_Type(Class)').values
X = unbalanced.values
_, smote_y = smote.fit_sample(X, y)
smote_target_count = pd.Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[ind_max_class]]

plt.figure()
multiple_bar_chart(plt.gca(), [target_count.index[ind_min_class], target_count.index[ind_max_class]],
                   values, 'Target', 'Frequency', 'Cover_Type(Class)')
plt.show()
