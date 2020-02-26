from functions import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as _stats


data = pd.read_csv('covtype.csv', sep=';')
np.random.seed(2)

dataSample = data.sample(frac=0.1)
# print("dataSample", dataSample)
dataSample = dataSample.astype({"W_A_1": 'category'})
dataSample = dataSample.astype({'W_A_2': 'category'})
dataSample = dataSample.astype({'W_A_3': 'category'})
dataSample = dataSample.astype({'W_A_4': 'category'})
dataSample = dataSample.astype({'Soil_Type_1': 'category'})
dataSample = dataSample.astype({'Soil_Type_2': 'category'})
dataSample = dataSample.astype({'Soil_Type_3': 'category'})
dataSample = dataSample.astype({'Soil_Type_4': 'category'})
dataSample = dataSample.astype({'Soil_Type_5': 'category'})
dataSample = dataSample.astype({'Soil_Type_6': 'category'})
dataSample = dataSample.astype({'Soil_Type_7': 'category'})
dataSample = dataSample.astype({'Soil_Type_8': 'category'})
dataSample = dataSample.astype({'Soil_Type_9': 'category'})
dataSample = dataSample.astype({'Soil_Type_10': 'category'})
dataSample = dataSample.astype({'Soil_Type_11': 'category'})
dataSample = dataSample.astype({'Soil_Type_12': 'category'})
dataSample = dataSample.astype({'Soil_Type_13': 'category'})
dataSample = dataSample.astype({'Soil_Type_14': 'category'})
dataSample = dataSample.astype({'Soil_Type_15': 'category'})
dataSample = dataSample.astype({'Soil_Type_16': 'category'})
dataSample = dataSample.astype({'Soil_Type_17': 'category'})
dataSample = dataSample.astype({'Soil_Type_18': 'category'})
dataSample = dataSample.astype({'Soil_Type_19': 'category'})
dataSample = dataSample.astype({'Soil_Type_20': 'category'})
dataSample = dataSample.astype({'Soil_Type_21': 'category'})
dataSample = dataSample.astype({'Soil_Type_22': 'category'})
dataSample = dataSample.astype({'Soil_Type_23': 'category'})
dataSample = dataSample.astype({'Soil_Type_24': 'category'})
dataSample = dataSample.astype({'Soil_Type_25': 'category'})
dataSample = dataSample.astype({'Soil_Type_26': 'category'})
dataSample = dataSample.astype({'Soil_Type_27': 'category'})
dataSample = dataSample.astype({'Soil_Type_28': 'category'})
dataSample = dataSample.astype({'Soil_Type_29': 'category'})
dataSample = dataSample.astype({'Soil_Type_30': 'category'})
dataSample = dataSample.astype({'Soil_Type_31': 'category'})
dataSample = dataSample.astype({'Soil_Type_32': 'category'})
dataSample = dataSample.astype({'Soil_Type_33': 'category'})
dataSample = dataSample.astype({'Soil_Type_34': 'category'})
dataSample = dataSample.astype({'Soil_Type_35': 'category'})
dataSample = dataSample.astype({'Soil_Type_36': 'category'})
dataSample = dataSample.astype({'Soil_Type_37': 'category'})
dataSample = dataSample.astype({'Soil_Type_38': 'category'})
dataSample = dataSample.astype({'Soil_Type_39': 'category'})
dataSample = dataSample.astype({'Soil_Type_40': 'category'})
dataSample = dataSample.astype({'Cover_Type(Class)': 'category'})
# print("dataSample", dataSample)


cols_nr = dataSample.select_dtypes(include='number')  # choose the numeric variables
# print("cols_nr", cols_nr)
# print("cols_nr.columns", cols_nr.columns)
cols_sb = dataSample.select_dtypes(include='category')  # choose categorical variables (there aren't any in the sample)
# print("cols_sb", cols_sb)
# print("cols_sb.columns", cols_sb.columns)
# NORMALIZATION
transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(cols_nr)
df_nr = pd.DataFrame(transf.transform(cols_nr), columns=cols_nr.columns)
# print("df_nr", df_nr)
df_sb = pd.DataFrame(cols_sb, columns=cols_sb.columns)
# print("df_sb", df_sb)
norm_data_minmax = df_nr.join(cols_sb, how='right')
# print(norm_data_minmax.describe(include='all'))
# print("norm_data_minmax", norm_data_minmax)
# norm_data_minmax2 = pd.concat([df_nr, df_sb], axis=1)
# norm_data_minmax2 = df_sb.join(df_nr, how='left')
# print(norm_data_minmax2.describe(include='all'))
# print("norm_data_minmax2", norm_data_minmax2)

fig, axs = plt.subplots(1, 2, figsize=(20, 10), squeeze=False)
axs[0, 0].set_title('Original data')
df_nr.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 1])
plt.show()

# df_nr.to_csv('df_nr.csv')
# df_sb.to_csv('df_sb.csv')

# OUTLIERS
# feito à mão em Excel

# FEATURE SELECTION
original = pd.read_csv('df_norm.csv', sep=',')
# print("original", original)
# print(original.describe(include='all'))

original = original.astype({"W_A_1": 'category'})
original = original.astype({'W_A_2': 'category'})
original = original.astype({'W_A_3': 'category'})
original = original.astype({'W_A_4': 'category'})
original = original.astype({'Soil_Type_1': 'category'})
original = original.astype({'Soil_Type_2': 'category'})
original = original.astype({'Soil_Type_3': 'category'})
original = original.astype({'Soil_Type_4': 'category'})
original = original.astype({'Soil_Type_5': 'category'})
original = original.astype({'Soil_Type_6': 'category'})
original = original.astype({'Soil_Type_7': 'category'})
original = original.astype({'Soil_Type_8': 'category'})
original = original.astype({'Soil_Type_9': 'category'})
original = original.astype({'Soil_Type_10': 'category'})
original = original.astype({'Soil_Type_11': 'category'})
original = original.astype({'Soil_Type_12': 'category'})
original = original.astype({'Soil_Type_13': 'category'})
original = original.astype({'Soil_Type_14': 'category'})
original = original.astype({'Soil_Type_15': 'category'})
original = original.astype({'Soil_Type_16': 'category'})
original = original.astype({'Soil_Type_17': 'category'})
original = original.astype({'Soil_Type_18': 'category'})
original = original.astype({'Soil_Type_19': 'category'})
original = original.astype({'Soil_Type_20': 'category'})
original = original.astype({'Soil_Type_21': 'category'})
original = original.astype({'Soil_Type_22': 'category'})
original = original.astype({'Soil_Type_23': 'category'})
original = original.astype({'Soil_Type_24': 'category'})
original = original.astype({'Soil_Type_25': 'category'})
original = original.astype({'Soil_Type_26': 'category'})
original = original.astype({'Soil_Type_27': 'category'})
original = original.astype({'Soil_Type_28': 'category'})
original = original.astype({'Soil_Type_29': 'category'})
original = original.astype({'Soil_Type_30': 'category'})
original = original.astype({'Soil_Type_31': 'category'})
original = original.astype({'Soil_Type_32': 'category'})
original = original.astype({'Soil_Type_33': 'category'})
original = original.astype({'Soil_Type_34': 'category'})
original = original.astype({'Soil_Type_35': 'category'})
original = original.astype({'Soil_Type_36': 'category'})
original = original.astype({'Soil_Type_37': 'category'})
original = original.astype({'Soil_Type_38': 'category'})
original = original.astype({'Soil_Type_39': 'category'})
original = original.astype({'Soil_Type_40': 'category'})
original = original.astype({'Cover_Type(Class)': 'category'})
# print("original", original)

feature_names = list(original.columns.values)
# print("feature_names", feature_names)
y_clf = original.pop('Cover_Type(Class)').values
X_clf = original.values
selector = SelectKBest(score_func=chi2, k=40)
features_df = selector.fit_transform(X_clf, y_clf)

# Get columns to keep and create new dataframe with those only
mask = selector.get_support()  # list of booleans
new_features = []  # The list of your K best features
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

dataframe = pd.DataFrame(features_df, columns=new_features)
dataframe['Cover_Type(Class)'] = y_clf
# print("dataframe", dataframe)
a = dataframe.describe(include='all')
print(a)
# print(dataframe.columns)
b = dataframe[['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
               'Horizontal_Distance_To_Fire_Points']].mean()
print(b)
# c = dataframe[['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
#                'Horizontal_Distance_To_Fire_Points']].quantile(q=.25)
# print("c", c)

# a.to_csv('describe_for_outliers.csv')

# dataframe.to_csv('df_FeatureSelection.csv', index=False, index_label=False)
