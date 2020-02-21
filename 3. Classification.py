from functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from subprocess import call
from sklearn.ensemble import GradientBoostingClassifier

# Training Strategy: hold-out (train_test_split): used in the presence of large thousands of records;
# Training method by order:
#           1.Naive Bayers
#           2.KNN
#           3.Decision Trees
#           4.Random Forests

# import the dataset into the dataframe, using pandas
data = pd.read_csv('covtype.csv', sep=';')
dataSample = data.sample(frac=0.05)

# Data preparation for the classification models
y: np.ndarray = dataSample.pop('Cover_Type(Class)').values
X: np.ndarray = dataSample.values
labels = pd.unique(y)
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)


# For printing data
def model_performance(tstY, prdY, n, d=' ', k=' ', n_name='n', d_name=' ', k_name=' '):
    #    import warnings
    #    warnings.filterwarnings('default')  # "error", "ignore", "always", "default", "module" or "once"
    accuracy = metrics.accuracy_score(tstY, prdY)
    precision = metrics.precision_score(tstY, prdY, average='macro')
    sensibility = metrics.recall_score(tstY, prdY, average='macro')
    print('Accurancy :', str(accuracy)[:6], \
          ' precision: ', str(precision)[:6], \
          ' sensibility: ' + str(sensibility)[:6],
          n_name, n, d_name, d, k_name, k)


# # Naive Bayes
# clf = GaussianNB()
# clf.fit(trnX, trnY)
# prdY = clf.predict(tstX)
# cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
# plt.figure()
# plot_confusion_matrix(plt.gca(), cnf_mtx, labels)
# plt.show()
# # Following part NOT working for unbalanced data due to negative values, uncomment for treted data testing
# # estimators = {'GaussianNB': GaussianNB(),
# #               'MultinomialNB': MultinomialNB(),
# #               'BernoulyNB': BernoulliNB()}
# # xvalues = []
# # yvalues = []
# # for clf in estimators:
# #     xvalues.append(clf)
# #     estimators[clf].fit(trnX, trnY)
# #     prdY = estimators[clf].predict(tstX)
# #     yvalues.append(metrics.accuracy_score(tstY, prdY))
# # plt.figure()
# # bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models', '', 'Accuracy', percentage=True)
# # plt.show()
#
# # KNN
# dataSample = data.sample(frac=0.5)
# nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# dist = ['Manhattan', 'Euclidean', 'Chebyshev']
# values = {}
# for d in dist:
#     yvalues = []
#     for n in nvalues:
#         knn = KNeighborsClassifier(n_neighbors=n, metric=d)
#         knn.fit(trnX, trnY)
#         prdY = knn.predict(tstX)
#         yvalues.append(metrics.accuracy_score(tstY, prdY))
#     values[d] = yvalues
# plt.figure()
# multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants', 'Number of Neighbors (K)', 'Accuracy', percentage=True)
# plt.show()
#
# # Decision Trees
# min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
# max_depths = [5, 10, 25, 50]
# criteria = ['entropy', 'gini']
#
# plt.figure()
# fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
# for k in range(len(criteria)):
#     f = criteria[k]
#     values = {}
#     for d in max_depths:
#         yvalues = []
#         for n in min_samples_leaf:
#             tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
#             tree.fit(trnX, trnY)
#             prdY = tree.predict(tstX)
#             yvalues.append(metrics.accuracy_score(tstY, prdY))
#             # print(metrics.accuracy_score(tstY, prdY))
#         values[d] = yvalues
#         multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
#                             'Number of Estimators'
#                             , 'Accuracy', percentage=True)
# plt.show()
#
# ## Following lines show the learned tree, using the graphviz package
# ## NOT WORKING
# # tree = DecisionTreeClassifier(max_depth=3)
# # tree.fit(trnX, trnY)
# #
# # dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
# # # Convert to png
# # call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png', '-Gdpi=600'])
# #
# # plt.figure(figsize=(14, 18))
# # plt.imshow(plt.imread('dtree.png'))
# # plt.axis('off')
# # plt.show()
#
# # Random Trees
# n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
# max_depths = [5, 10, 25, 50]
# max_features = ['Square Root', 'logarithm base 2']
#
# plt.figure()
# fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
# for k in range(len(max_features)):
#     f = max_features[k]
#     values = {}
#     for d in max_depths:
#         yvalues = []
#         for n in n_estimators:
#             rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
#             rf.fit(trnX, trnY)
#             prdY = rf.predict(tstX)
#             yvalues.append(metrics.accuracy_score(tstY, prdY))
#         values[d] = yvalues
#     multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f, 'Number of Estimators',
#                         'Accuracy', percentage=True)
# plt.show()
#

# # Gradient Boosting
# def GB(data1, save=None):
#     from sklearn.ensemble import GradientBoostingClassifier
#     print('Gradient Boosting Started \n')
#       # Existing error in Cover_Type(Class)
#     y: np.ndarray = data1.pop('Cover_Type(Class)').values
#     X: np.ndarray = data1.values
#     labels = pd.unique(y)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)
#
#     estimators = [5, 50, 100, 150, 200, 300]
#     l_rate = [0.1]
#     max_depth = [2, 10, 20]
#
#     yvalues = []
#     for l in range(len(l_rate)):
#         values = {}
#         for d in range(len(max_depth)):
#             yvalues = []
#             for e in range(len(estimators)):
#                 gbc = GradientBoostingClassifier(n_estimators=estimators[e], learning_rate=l_rate[l],
#                                                  max_depth=max_depth[d])
#                 gbc.fit(X_train, y_train)
#                 gbc.score(X_test, y_test)
#                 prdY = gbc.predict(X_test)
#
#                 yvalues.append(metrics.accuracy_score(y_test, prdY))
#                 model_performance(y_test, prdY, estimators[e], max_depth[d], l_rate[l], n_name='  Estim:',
#                                   d_name='depth:', k_name='l_rate:')  # prints the data of the graph
#         values[d] = yvalues
#         plt.figure()
#         multiple_line_chart(plt.gca(), estimators, values, 'Gradient Boosting estimators', 'n', 'accuracy',
#                             percentage=False)  # changed to false
#         if save == 'p':
#             plt.savefig('Gradient_Boosting.png')
#         plt.show()
#
#
# GB(dataSample, 'p')
