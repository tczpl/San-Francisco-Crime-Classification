import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils

from copy import deepcopy
from datetime import datetime
from matplotlib.colors import LogNorm

trainDF = pd.read_csv("../input/train.csv")

xy_scaler = preprocessing.StandardScaler()
xy_scaler.fit(trainDF[["X","Y"]])
trainDF[["X","Y"]] = xy_scaler.transform(trainDF[["X","Y"]])
trainDF = trainDF[abs(trainDF["Y"]) < 100]
trainDF.index = range(len(trainDF))

def parse_time(x):
    DD = datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    time = DD.hour
    day = DD.day
    month = DD.month
    year = DD.year
    return time,day,month,year

def get_season(x):
    summer = 0
    fall = 0
    winter = 0
    spring = 0
    if (x in [5, 6, 7]):
        summer = 1
    if (x in [8, 9, 10]):
        fall = 1
    if (x in [11, 0, 1]):
        winter = 1
    if (x in [2, 3, 4]):
        spring = 1
    return summer,fall,winter,spring

def parse_data(df,logodds,logoddsPA):
    feature_list = df.columns.tolist()
    if "Descript" in feature_list:
        feature_list.remove("Descript")
    if "Resolution" in feature_list:
        feature_list.remove("Resolution")
    if "Category" in feature_list:
        feature_list.remove("Category")
    if "Id" in feature_list:
        feature_list.remove("Id")
    data2 = df[feature_list]
    data2.index = range(len(df))
    print("Creating address features")
    address_features = data2["Address"].apply(lambda x: logodds[x])
    address_features.columns = ["logodds" + str(x) for x in range(len(address_features.columns))]
    print("Parsing dates")
    data2["Time"], data2["Day"], data2["Month"], data2["Year"] = zip(*data2["Dates"].apply(parse_time))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print("Creating one-hot variables")
    dummy_ranks_PD = pd.get_dummies(data2['PdDistrict'], prefix = 'PD')
    dummy_ranks_DAY = pd.get_dummies(data2["DayOfWeek"], prefix = 'DAY')
    data2["IsInterection"] = data2["Address"].apply(lambda x: 1 if "/" in x else 0)
    data2["logoddsPA"] = data2["Address"].apply(lambda x: logoddsPA[x])
    print("droping processed columns")
    data2 = data2.drop("PdDistrict",axis = 1)
    data2 = data2.drop("DayOfWeek",axis = 1)
    data2 = data2.drop("Address",axis = 1)
    data2 = data2.drop("Dates",axis = 1)
    feature_list = data2.columns.tolist()
    print("joining one-hot features")
    features = data2[feature_list].join(dummy_ranks_PD.ix[:,:]).join(dummy_ranks_DAY.ix[:,:]).join(address_features.ix[:,:])
    print("creating new features")
    features["IsDup"] = pd.Series(features.duplicated() | features.duplicated(take_last = True)).apply(int)
    features["Awake"] = features["Time"].apply(lambda x: 1 if (x == 0 or (x >= 8 and x <= 23)) else 0)
    features["Summer"], features["Fall"], features["Winter"], features["Spring"] = zip(*features["Month"].apply(get_season))
    if "Category" in df.columns:
        labels = df["Category"].astype('category')
    else:
        labels = None
    return features,labels

addresses = sorted(trainDF["Address"].unique())
categories = sorted(trainDF["Category"].unique())
C_counts = trainDF.groupby(["Category"]).size()
A_C_counts = trainDF.groupby(["Address","Category"]).size()
A_counts = trainDF.groupby(["Address"]).size()
logodds = {}
logoddsPA = {}
MIN_CAT_COUNTS = 2
default_logodds = np.log(C_counts / len(trainDF)) - np.log(1.0 - C_counts / float(len(trainDF)))
for addr in addresses:
    PA = A_counts[addr] / float(len(trainDF))
    logoddsPA[addr] = np.log(PA) - np.log(1.0 - PA)
    logodds[addr] = deepcopy(default_logodds)
    for cat in A_C_counts[addr].keys():
        if (A_C_counts[addr][cat] > MIN_CAT_COUNTS) and A_C_counts[addr][cat] < A_counts[addr]:
            PA = A_C_counts[addr][cat] / float(A_counts[addr])
            logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0 - PA)
    logodds[addr] = pd.Series(logodds[addr])
    logodds[addr].index = range(len(categories))

features, labels = parse_data(trainDF,logodds,logoddsPA)

print(features.columns.tolist())
print(len(features.columns))

collist = features.columns.tolist()
scaler = preprocessing.StandardScaler()
scaler.fit(features)
features[collist] = scaler.transform(features)

new_PCA = PCA(n_components = 60)
new_PCA.fit(features)
print(new_PCA.explained_variance_ratio_)

sss = StratifiedShuffleSplit(labels, train_size = 0.5)
for train_index, test_index in sss:
    features_train,features_test = features.iloc[train_index],features.iloc[test_index]
    labels_train,labels_test = labels[train_index],labels[test_index]
features_test.index = range(len(features_test))
features_train.index = range(len(features_train))
labels_train.index = range(len(labels_train))
labels_test.index = range(len(labels_test))
features.index = range(len(features))
labels.index = range(len(labels))

def build_and_fit_model(X_train,y_train,X_test = None,y_test = None,hn = 32,dp = 0.5,layers = 1,epochs = 1,batches = 64,verbose = 0):
    input_dim = X_train.shape[1]
    output_dim = len(labels_train.unique())
    Y_train = np_utils.to_categorical(y_train.cat.rename_categories(range(len(y_train.unique()))))
    model = Sequential()
    model.add(Dense(hn,input_shape = (input_dim,)))
    model.add(PReLU())
    model.add(Dropout(dp))

    for i in range(layers):
      model.add(Dense(hn))
      model.add(PReLU())
      model.add(BatchNormalization())
      model.add(Dropout(dp))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    
    if X_test is not None:
        Y_test = np_utils.to_categorical(y_test.cat.rename_categories(range(len(y_test.unique()))))
        fitting = model.fit(X_train, Y_train, nb_epoch = epochs, batch_size = batches, verbose = verbose, validation_data = (X_test,Y_test))
        test_score = log_loss(y_test, model.predict_proba(X_test,verbose = 0))
    else:
        model.fit(X_train, Y_train, nb_epoch = epochs, batch_size = batches, verbose=verbose)
        fitting = 0
        test_score = 0
    return test_score, fitting, model

len(features.columns)

N_EPOCHS = 80
N_HN = 600
N_LAYERS = 1
DP = 0.5

#score, fitting, model = build_and_fit_model(features_train.as_matrix(),labels_train,X_test = features_test.as_matrix(),y_test = labels_test,hn = N_HN,layers = N_LAYERS,epochs = N_EPOCHS,verbose = 2,dp = DP)

#model = LogisticRegression()
#model.fit(features_train,labels_train)

score, fitting, model = build_and_fit_model(features.as_matrix(),labels,hn = N_HN,layers = N_LAYERS,epochs = N_EPOCHS,verbose = 2,dp = DP)

print("all", log_loss(labels, model.predict_proba(features.as_matrix())))

testDF = pd.read_csv("../input/test.csv")
testDF[["X","Y"]] = xy_scaler.transform(testDF[["X","Y"]])
testDF["X"] = testDF["X"].apply(lambda x: 0 if abs(x) > 5 else x)
testDF["Y"] = testDF["Y"].apply(lambda y: 0 if abs(y) > 5 else y)

new_addresses = sorted(testDF["Address"].unique())
new_A_counts = testDF.groupby("Address").size()
only_new = set(new_addresses + addresses) - set(addresses)
only_old = set(new_addresses + addresses) - set(new_addresses)
in_both = set(new_addresses).intersection(addresses)
for addr in only_new:
    PA = new_A_counts[addr] / float(len(testDF) + len(trainDF))
    logoddsPA[addr] = np.log(PA) - np.log(1.0 - PA)
    logodds[addr] = deepcopy(default_logodds)
    logodds[addr].index = range(len(categories))
for addr in in_both:
    PA = (A_counts[addr] + new_A_counts[addr]) / float(len(testDF) + len(trainDF))
    logoddsPA[addr] = np.log(PA) - np.log(1.0 - PA)

features_sub, _ = parse_data(testDF,logodds,logoddsPA)

collist = features_sub.columns.tolist()
print(collist)

features_sub[collist] = scaler.transform(features_sub[collist])

predDF = pd.DataFrame(model.predict_proba(features_sub.as_matrix()),columns = sorted(labels.unique()))

predDF.head()

predDF.to_csv("submission_1x600x80_0.5.csv")