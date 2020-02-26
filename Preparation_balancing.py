from functions import *
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder
from imblearn.over_sampling import SMOTE

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('df_FeatureSelection.csv', sep=',')
np.random.seed(2)

dataSample = data
unbalanced = dataSample
target_count = unbalanced['Cover_Type(Class)'].value_counts()
plt.figure()
plt.title('Cover_Type(Class) balance')
plt.bar(target_count.index, target_count.values)
plt.show()

min_class = target_count.idxmin()  # identify the index for the class with the minimum number of elements (class 4)
max_class = target_count.idxmax()  # identify the index for the class with the maximum number of elements (class 2)

class1 = target_count[1]
# print("Class1 contains these many elements:", class1)
class2 = target_count[2]
# print("Class2 contains these many elements:", class2)
class3 = target_count[3]
# print("Class3 contains these many elements:", class3)
class4 = target_count[4]
# print("Class4 contains these many elements:", class4)
class5 = target_count[5]
# print("Class5 contains these many elements:", class5)
class6 = target_count[6]
# print("Class6 contains these many elements:", class6)
class7 = target_count[7]
# print("Class7 contains these many elements:", class7)

print('Minority class:', target_count[min_class])
# print('Test class:', target_count[3])
print('Majority class:', target_count[max_class])
print('Proportion:', round(target_count[min_class] / target_count[max_class], 2), ': 1')

ind_max_class = target_count.index.get_loc(max_class)  # class 2
# print("ind_max_class", ind_max_class)
ind_min_class = target_count.index.get_loc(min_class)  # class 4
# print("ind_min_class", ind_min_class)

# ind_class1 = 1
# print("ind_class1", ind_class1)
# print("target_count.values[ind_class1]", target_count.values[ind_class1])
# ind_class2 = ind_max_class
# ind_class2 = 2
# print("ind_class2", ind_class2)
# print("target_count.values[ind_class2]", target_count.values[ind_class2])
# ind_class3 = 3
# print("target_count.values[ind_class3]", target_count.values[ind_class3])
# # ind_class4 = ind_min_class
# ind_class4 = 4
# print("ind_class4", ind_class4)
# print("target_count.values[ind_class4]", target_count.values[ind_class4])
# ind_class5 = 5
# print("target_count.values[ind_class5]", target_count.values[ind_class5])
# ind_class6 = 6
# print("target_count.values[ind_class6]", target_count.values[ind_class6])
# ind_class7 = 7
# print("ind_class7", ind_class7)
# print("target_count.values[ind_class7]", target_count.values[ind_class7])

# (28) split the dataset into subdatasets, one for each class of the class variable (Cover Type, in this case)
# values = {'Original': [target_count.values[ind_class1], target_count.values[ind_class2],
#                        target_count.values[ind_class3], target_count.values[ind_class4],
#                        target_count.values[ind_class5], target_count.values[ind_class6]]}
# values = {'Original': [target_count.values[ind_class2], target_count.values[ind_class1],
#                        target_count.values[ind_class3], target_count.values[ind_class7],
#                        target_count.values[ind_class6], target_count.values[ind_class5],
#                        target_count.values[ind_class4]]}
values = {'Original': [target_count.values[1], target_count.values[0],
                       target_count.values[2], target_count.values[6],
                       target_count.values[5], target_count.values[4],
                       target_count.values[3]]}
# print("values Original", values)

# creating the classes that are to be oversampled, instead of min and max from the teacher's example
df_class1 = unbalanced[unbalanced['Cover_Type(Class)'] == 1]
# print("df_class1", df_class1)
df_class2 = unbalanced[unbalanced['Cover_Type(Class)'] == 2]
# print("df_class2", df_class2)
df_class3 = unbalanced[unbalanced['Cover_Type(Class)'] == 3]
# print("df_class3", df_class3)
df_class4 = unbalanced[unbalanced['Cover_Type(Class)'] == 4]
# print("df_class4", df_class4)
df_class5 = unbalanced[unbalanced['Cover_Type(Class)'] == 5]
# print("df_class5", df_class5)
df_class6 = unbalanced[unbalanced['Cover_Type(Class)'] == 6]
# print("df_class6", df_class6)
df_class7 = unbalanced[unbalanced['Cover_Type(Class)'] == 7]
# print("df_class7", df_class7)


# df_class_min = unbalanced[unbalanced['Cover_Type(Class)'] == min_class]  # DataFrame
df_class_min = df_class4
# df_class_max = unbalanced[unbalanced['Cover_Type(Class)'] == max_class]
df_class_max = df_class2

# df_under = df_class_max.sample(len(df_class_min))
# df_under = df_class4
df_class1_under = df_class1.sample(len(df_class4))
# print("df_class1_under", df_class1_under)
df_class2_under = df_class2.sample(len(df_class4))
df_class3_under = df_class3.sample(len(df_class4))
df_class4_under = df_class4
df_class5_under = df_class5.sample(len(df_class4))
df_class6_under = df_class6.sample(len(df_class4))
df_class7_under = df_class7.sample(len(df_class4))
under_frames = [df_class2_under, df_class3_under, df_class4_under, df_class5_under, df_class6_under, df_class7_under]
df_under = df_class1_under.append(under_frames)
# print("df_under", df_under)
# values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]
values['UnderSample'] = [target_count.values[6], target_count.values[6], target_count.values[6],
                         target_count.values[6], target_count.values[6], target_count.values[6],
                         target_count.values[6]]
# print("values Original + UnderSample", values)

# df_over = df_class_min.sample(len(df_class_max), replace=True)
# df_over = df_class2
df_class1_over = df_class1.sample(len(df_class2), replace=True)
# print("df_class1_over", df_class1_over)
df_class2_over = df_class2
df_class3_over = df_class3.sample(len(df_class2), replace=True)
df_class4_over = df_class4.sample(len(df_class2), replace=True)
df_class5_over = df_class5.sample(len(df_class2), replace=True)
df_class6_over = df_class6.sample(len(df_class2), replace=True)
df_class7_over = df_class7.sample(len(df_class2), replace=True)
over_frames = [df_class2_over, df_class3_over, df_class4_over, df_class5_over, df_class6_over, df_class7_over]
df_over = df_class1_over.append(over_frames)
# print("df_over", df_over)

# values['OverSample'] = [len(df_over), target_count.values[ind_max_class]]
values['OverSample'] = [target_count.values[0], target_count.values[0], target_count.values[0],
                        target_count.values[0], target_count.values[0], target_count.values[0],
                        target_count.values[0]]
# print("values Original + UnderSample + OverSample", values)

smote = SMOTE(random_state=42)
# print("smote", smote)
y = unbalanced.pop('Cover_Type(Class)').values
X = unbalanced.values
_, smote_y = smote.fit_sample(X, y)
# print("_", _)
# print("smote_y", smote_y)
smote_target_count = pd.Series(smote_y).value_counts()
smote_target_count.describe()
df_smote = pd.DataFrame(_, smote_y)
# print("df_smote", df_smote)
# values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[ind_max_class]] # 7 classes
values['SMOTE'] = [smote_target_count.values[0], smote_target_count.values[1],
                   smote_target_count.values[2], smote_target_count.values[3],
                   smote_target_count.values[4], smote_target_count.values[5],
                   smote_target_count.values[6]]  # 7 classes
# print("values Original + UnderSample + OverSample + SMOTE", values)

plt.figure()
# multiple_bar_chart(plt.gca(),
#                         [target_count.index[ind_min_class], target_count.index[1-ind_min_class]],
#                         values, 'Target', 'frequency', 'Class balance')
multiple_bar_chart(plt.gca(), [target_count.index[0], target_count.index[1],
                               target_count.index[2], target_count.index[3],
                               target_count.index[4], target_count.index[5],
                               target_count.index[6]],
                   values, 'Target', 'Frequency', 'Cover_Type(Class)')
plt.show()


# EXPORTS TO CSV FILES

# df_smote.to_csv('SMOTE_sample.csv')
# df_under.to_csv('Undersample.csv')
# df_over.to_csv('Oversample.csv')
