import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import datetime

train = pd.read_csv("Predict_NoShow_Train.csv", index_col="ID")


######## outlier detection################
# =============================================================================
# from sklearn.ensemble import IsolationForest
# 
# out_detect_data = train.loc[:, ['Age', 'DaysUntilAppointment']]
# # fit the model
# clf = IsolationForest(max_samples=100)
# clf.fit(out_detect_data)
# y_pred_train = clf.predict(out_detect_data)
# 
# inlier_idx = y_pred_train != -1
# train = train.iloc[inlier_idx, :]
# =============================================================================
###########################################


X_label = (train['Status'] != "Show-Up")*1
X_train = train.drop(['Status'], axis=1)

#feature scaling on numeric predictors
scaler = preprocessing.StandardScaler().fit(train.loc[:, ['Age', 'DaysUntilAppointment']])


def pre_process(data):
    data['Handicapped'] = data['Handicapped'].astype('category')
    data['RemindedViaSMS'] = data['RemindedViaSMS'].astype('category')
    
    # get month
    
    data['AppointmentMadeMonth'] = [d8[5:7] for d8 in data.loc[:,'DateAppointmentWasMade'] ]
    data['AppointmentMonth'] = [d8[5:7] for d8 in data.loc[:,'DateOfAppointment']]
    
    # get year 
    
    data['AppointmentMadeYear'] = [d8[0:4] for d8 in data.loc[:,'DateAppointmentWasMade']]
    data['AppointmentYear'] = [d8[0:4] for d8 in data.loc[:,'DateOfAppointment']]
    
    
    # get day for the month 
    
    data['AppointmentMadeDay'] = [d8[8:10] for d8 in data.loc[:,'DateAppointmentWasMade']]
    data['AppointmentMDay'] = [d8[8:10] for d8 in data.loc[:,'DateOfAppointment']]
    dayAM_of_week = [datetime.datetime(int(row['AppointmentMadeYear']), int(row['AppointmentMadeMonth']),
                                       int(row['AppointmentMadeDay'])) for index, row in data.iterrows()]
       
    # convert into day of the week
    
    dayAM_of_week = [_.weekday() for _ in dayAM_of_week]
    dayAM_of_week_dict = {0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    data['AppointmentMadeDayofWeek'] = dayAM_of_week
    data = data.replace({'AppointmentMadeDayofWeek': dayAM_of_week_dict})
    
    # get rid of original date variables
    
    del data['DateAppointmentWasMade'], data['DateOfAppointment']
        
    #encode dummy variables
    X_data_d = pd.get_dummies(data)

    X_data_d.loc[:, ['Age', 'DaysUntilAppointment']] = scaler.transform(X_data_d.loc[:,['Age', 'DaysUntilAppointment']])

    return X_data_d

X_train_d = pre_process(train)


#####feature selection########
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold


xgb = XGBClassifier(objective ='binary:logistic', n_jobs = -1)
rfecv = RFECV(estimator=xgb, step=1, cv=StratifiedKFold(2),
              scoring='neg_log_loss', n_jobs = -1)
rfecv.fit(X_train_d, X_label)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (log loss)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), -rfecv.grid_scores_)
plt.show()

#X_train_s = rfecv.transform(X_train_d)

rfe = RFE(xgb, n_features_to_select= 77)
rfe.fit(X_train_d, X_label)
#print(rfe.ranking_)
idx = rfe.ranking_ == 1
X_train_s = X_train_d.iloc[:,idx]


colnames_train = [col for col in X_train_d]
from itertools import compress
colnames_select = list(compress(colnames_train, idx))



############Logistic Regression####################
from sklearn.linear_model import LogisticRegression

#grid search on cross validation 
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

C_grid = {'C': [ 0.2, 0.25, 0.03, 0.035] }
logistic_grid = GridSearchCV(LogisticRegression(solver = 'sag', max_iter= 500),  C_grid, scoring= 'neg_log_loss',  n_jobs = -1)
logistic_grid.fit(X_train_s, X_label)

best_random = logistic_grid.best_estimator_
print(logistic_grid.best_estimator_)


########### Random Forest#############
from sklearn.ensemble import RandomForestClassifier

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [ 13, 15, 17, 19],
    'max_features': [11, 13, 15, 17],
    'min_samples_leaf': [1],
    'min_samples_split': [10],
    'n_estimators': [140, 145, 150, 155, 160]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring= 'neg_log_loss')

# Fit the grid search to the data
grid_search_rf.fit(X_train_d, X_label)
best_grid_rf = grid_search_rf.best_estimator_



############ Neural Network#######################
from sklearn.neural_network import MLPClassifier


param_grid = {
    'hidden_layer_sizes': [(6,), (7,), (8,), (9,), (10,), (11,)],
    'alpha': [0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25],
}

neural = MLPClassifier()
grid_search_nn = GridSearchCV(estimator = neural, param_grid = param_grid,
                              cv = 3, n_jobs = -1, verbose = 2, scoring= 'neg_log_loss')

# Fit the grid search to the data
grid_search_nn.fit(X_train_s, X_label)

best_grid_nn = grid_search_nn.best_estimator_



############ Xgboost #########################
from xgboost import XGBClassifier
param_grid = {
    'n_estimators' : [100, 120, 140, 160, 180],
    'max_depth' : [5, 7, 9, 11],   
    'subsample' : [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_lambda' :[0.05, 0.1, 0.15, 0.2],
}
  

xgb = XGBClassifier(objective ='binary:logistic', n_jobs = -1)

grid_search_xgb = GridSearchCV(estimator = xgb, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring= 'neg_log_loss')

# Fit the grid search to the data
grid_search_xgb.fit(X_train_s, X_label)

best_grid_xgb = grid_search_xgb.best_estimator_


############# Linear SVC ##########################
from sklearn.svm import LinearSVC
svc = LinearSVC(random_state=0)

param_grid = {
    'C' : [0.01, 0.03, 0.05, 0.07, 0.09 ],
}


grid_search_svc = GridSearchCV(estimator = svc, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_svc.fit(X_train_s, X_label)
best_grid_svc = grid_search_svc.best_estimator_



######################### LDA ####################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver ='lsqr')
param_grid = {
    'shrinkage' : [0.01, 0.015, 0.018, 0.02, 0.025, 0.03],
}
grid_search_lda = GridSearchCV(estimator = lda, param_grid = param_grid, 
                          cv = 3, n_jobs = -1)
grid_search_lda.fit(X_train_s, X_label)
best_grid_lda = grid_search_lda.best_estimator_
print(best_grid_lda)



######################### QDA ####################
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis( )
param_grid = {
    'reg_param' : [ 1.75, 1.8, 1.85],
}
grid_search_qda = GridSearchCV(estimator = qda, param_grid = param_grid, 
                          cv = 3, n_jobs = -1)
grid_search_qda.fit(X_train_s, X_label)
best_grid_qda = grid_search_qda.best_estimator_
print(best_grid_qda)


################### Adaboost ########################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

param_grid = {"n_estimators" : [50, 80, 100, 120],
              "base_estimator__max_depth": [10],
              "base_estimator__max_features": [ 15],
             }


DTC = DecisionTreeClassifier( max_depth = 10, max_features = 15,  random_state = 11)

ABC = AdaBoostClassifier(base_estimator = DTC)

# run grid search
grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'neg_log_loss',
                               n_jobs = -1)
grid_search_ABC.fit(X_train_s, X_label)
best_grid_ABC = grid_search_ABC.best_estimator_
print(best_grid_ABC)



