# LAB4 - EVALUATION

# The evaluation of the results of each learnt model, in the classification paradigm, 
# is objective and straightforward. We just need to assess if the predicted labels are 
# correct, which is done by measuring the number of records where the predicted label is 
# equal to the known ones.

# ACCURACY
# (6)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('data/diabetes.csv')  # after training
y = data.pop('class').values
X = data.values

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

clf = GaussianNB()
clf.fit(trnX, trnY)
clf.score(tstX, tstY)

# CONFUSION MATRIX
# (7) - array for binary variable
import numpy as np
import sklearn.metrics as metrics

labels: np.ndarray = pd.unique(y)
prdY: np.ndarray = clf.predict(tstX)
cnf_mtx: np.ndarray = metrics.confusion_matrix(tstY, prdY, labels)
cnf_mtx

# (8) - matrix chart for binary variable (non-normalized and normalized)
import itertools
import matplotlib.pyplot as plt

CMAP = plt.cm.Blues


# ---------- the function
def plot_confusion_matrix(ax: plt.Axes, cnf_matrix: np.ndarray, classes_names: list, normalize: bool = False):
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix'
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=CMAP)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")


# ----------

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
plot_confusion_matrix(axs[0, 0], cnf_mtx, labels)
plot_confusion_matrix(axs[0, 1], metrics.confusion_matrix(tstY, prdY, labels), labels, normalize=True)
plt.tight_layout()
plt.show()

# (9) - matrix chart for any variable (non-normalized and normalized)
data = pd.read_csv('data/iris.csv')
y = data.pop('class').values
X = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

clf = GaussianNB()
clf.fit(trnX, trnY)
prdY = clf.predict(tstX)

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
plot_confusion_matrix(axs[0, 0], metrics.confusion_matrix(tstY, prdY, labels), labels)
plot_confusion_matrix(axs[0, 1], metrics.confusion_matrix(tstY, prdY, labels), labels, normalize=True)
plt.tight_layout()
plt.show()


# ROC CHARTS
# (10)
def plot_roc_chart(ax: plt.Axes, models: dict, tstX, tstY, target: str = 'class'):
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('FP rate')
    ax.set_ylabel('TP rate')
    ax.set_title('ROC chart for %s' % target)
    ax.plot([0, 1], [0, 1], color='navy', label='random', linestyle='--')

    for clf in models:
        scores = models[clf].predict_proba(tstX)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(tstY, scores, 'positive')
        roc_auc = metrics.roc_auc_score(tstY, scores)
        ax.plot(fpr, tpr, label='%s (auc=%0.2f)' % (clf, roc_auc))
    ax.legend(loc="lower center")


data = pd.read_csv('data/diabetes.csv')
y = data.pop('class').values
X = data.values
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
model = GaussianNB().fit(trnX, trnY)

plt.figure()
plot_roc_chart(plt.gca(), {'GaussianNB': model}, tstX, tstY, 'class')
plt.show()

# to obtain the area under the roc curve: roc_auc_score command, available in the 
# sklearn.metrics package