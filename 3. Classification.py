from functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import ipynb.fs.defs.functions as func
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

# # Naive Bayes

# clf = GaussianNB()
# clf.fit(trnX, trnY)
# prdY = clf.predict(tstX)
# cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
# plt.figure()
# func.plot_confusion_matrix(plt.gca(), cnf_mtx, labels)
# plt.show()

# Not working ...
# estimators = {'GaussianNB': GaussianNB(),
#               'MultinomialNB': MultinomialNB(),
#               'BernoulyNB': BernoulliNB()}
# xvalues = []
# yvalues = []
# for clf in estimators:
#     xvalues.append(clf)
#     estimators[clf].fit(trnX, trnY)
#     prdY = estimators[clf].predict(tstX)
#     yvalues.append(metrics.accuracy_score(tstY, prdY))
# plt.figure()
# func.bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
# plt.show()
#

# # KNN
#
# dataSample = data.sample(frac=0.5)
# nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# dist = ['manhattan', 'euclidean', 'chebyshev']
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
# func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants', 'n', 'accuracy', percentage=True)
# plt.show()

# Decision Trees

min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
max_depths = [5, 10, 25, 50]
criteria = ['entropy', 'gini']

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for n in min_samples_leaf:
            tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
            # print(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues
        func.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                                 'nr estimators'
                                 , 'accuracy', percentage=True)
plt.show()
