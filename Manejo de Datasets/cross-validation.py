import random
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.utils import resample

X = ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"]

# ~ #Shuffle
random.shuffle(X)
print ('Shuffle {}'.format(X))
print ('----------------------')


#Leave One Out
print ('LeaveOneOut')
loo = LeaveOneOut()
print (loo.get_n_splits(X))
for train, test in loo.split(X):
	print("train {} test {}".format(train, test))
print ('----------------------')

# ~ #k-fold cross validation
print ('k-fold cross validation')
k=5
print('k =',k)
kf = KFold(n_splits=k)
for train, test in kf.split(X):
	print("train {} test {}".format(train, test))
print ('----------------------')

#Bootstrap
print ('Bootstrap')
print('X', *X)
train = resample(X, n_samples = len(X))
print ('train {}'.format(train))
test = np.array([x for x in X if x not in train]) 
print ('test {}'.format(test))














