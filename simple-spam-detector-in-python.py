import pandas as pd
from sklearn.naive_bayes import MultinomialNB as nb #importing naive-bayes impl.
import numpy as np 


#download this dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/
#store it in a local directory
#import dataset with pandas read_csv() method
dataset=pd.read_csv("spambase.data").as_matrix()

#shuffle dataset rows in order to make it more random for to split it into train & test datasets
np.random.shuffle(dataset)


#define set of independent attributes, first 48 cols
X=dataset[:,:48]

#define dependent attribute/target, last column
Y=dataset[:,-1]

#define training set

X_train=X[:-100,:]
Y_train=Y[:-100,]

#define test set

X_test=X[-100:,]
Y_test=Y[-100:,]

print("Checkpoint I")
#create a model instance of naive-bayes

naive_instance=nb()

print("Checkpoint II")

naive_instance.fit(X_train,Y_train)
print("Classification Score for Naive-Bayes is -:",naive_instance.score(X_test,Y_test))


print("Checkpoint III")

from sklearn.ensemble import AdaBoostClassifier as ABC 
#create a model instance of AdaBoost
adaboost=ABC()
adaboost.fit(X_train,Y_train)
print("Classification Score for AdaBoost -: ",adaboost.score(X_test,Y_test))

print("Checkpoint IV")


from sklearn.ensemble import RandomForestClassifier as rf 
#create a model instance of RandomForest
clf = rf()
clf.fit(X_train, Y_train)
print("Classification Score for RandomForestClassifier -: ",clf.score(X_test,Y_test))


from sklearn.ensemble import ExtraTreesClassifier as etc
#create a model instance of ExtraTreesClassifier
extratrees = etc()
extratrees.fit(X_train, Y_train)
print("Classification Score for ExtraTreesClassifier -: ",extratrees.score(X_test,Y_test))

print("Checkpoint V")

from sklearn.tree import DecisionTreeClassifier as dtc
#create a model instance of DecisionTreeClassifier
dectrees = dtc()
dectrees.fit(X_train, Y_train)
print("Classification Score for DecisionTreeClassifier -: ",dectrees.score(X_test,Y_test))
