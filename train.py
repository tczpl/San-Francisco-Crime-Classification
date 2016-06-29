import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
# from keras.layers.advanced_activations import PReLU
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.normalization import BatchNormalization
# from keras.models import Sequential
# from keras.utils import np_utils
from copy import deepcopy


features=pd.read_csv("features.csv")
del features[features.columns[0]]
labelsdf=pd.read_csv("labels.csv")
del labelsdf[labelsdf.columns[0]]

labels = labelsdf["labels"].astype('category')

features_sub=pd.read_csv("features_sub.csv")
del features_sub[features_sub.columns[0]]


print ("gogogo")
model = RandomForestClassifier(n_estimators=10, min_samples_split=4, verbose=True, n_jobs=4, max_features=None)
model.fit(features,labels)

#print("all", log_loss(labels, model.predict_proba(features.as_matrix())))

predDF=pd.DataFrame(model.predict_proba(features_sub.as_matrix()),columns=sorted(labels.unique()))


# In[37]:

predDF.head()

# In[38]:

#	import gzip
#	with gzip.GzipFile('submission.csv.gz',mode='w',compresslevel=9) as gzfile:
#		predDF.to_csv(gzfile,index_label="Id",na_rep="0")

predDF.to_csv("rf_666.csv")