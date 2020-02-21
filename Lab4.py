import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('covtype.csv', sep=';')
dataSample = data.sample(frac=0.1)
y = dataSample.pop('Cover_Type(Class)').values
X = dataSample.values

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

clf = GaussianNB()
clf.fit(trnX, trnY)
clf.score(tstX, tstY)
