import pickle

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load the data

iris=datasets.load_iris()

x=iris.data[:,:2]

y=iris.target

# train model

clf=RandomForestClassifier(max_depth=2, n_estimators=5, random_state=47)

clf.fit(x,y)

# Save model

with open("clf.pickle", "wb") as f:
	pickle.dump(clf,f)
