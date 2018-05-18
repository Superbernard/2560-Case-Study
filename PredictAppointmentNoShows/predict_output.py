def authors():
    print("Zhicong Chu", "Bowei Wei", "Ning Zhang", "Jiarou Quan")



def predictNoshow():
    
    import pandas as pd 
    import numpy as np 
    from sklearn import preprocessing
    from sklearn.model_selection import cross_val_score
    import datetime

    train = pd.read_csv("Predict_NoShow_Train.csv", index_col="ID")
    
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
    
    
    ##### after feature selection########

    colnames_select = ['Age', 'DaysUntilAppointment', 'Diabetes', 'Alcoholism', 
                       'Hypertension', 'Smoker', 'Scholarship', 'Tuberculosis', 
                       'Gender_F', 'Handicapped_0', 'Handicapped_1', 'RemindedViaSMS_0', 
                       'RemindedViaSMS_1', 'RemindedViaSMS_2', 'DayOfTheWeek_Friday',
                       'DayOfTheWeek_Monday', 'DayOfTheWeek_Saturday', 'DayOfTheWeek_Tuesday',
                       'DayOfTheWeek_Wednesday', 'AppointmentMadeMonth_01', 
                       'AppointmentMadeMonth_02', 'AppointmentMadeMonth_05', 
                       'AppointmentMadeMonth_07', 'AppointmentMadeMonth_08', 
                       'AppointmentMadeMonth_09', 'AppointmentMadeMonth_11',
                       'AppointmentMadeMonth_12', 'AppointmentMonth_01', 
                       'AppointmentMonth_02', 'AppointmentMonth_03', 'AppointmentMonth_05',
                       'AppointmentMonth_06', 'AppointmentMonth_09', 'AppointmentMonth_10',
                       'AppointmentMonth_11', 'AppointmentMonth_12', 'AppointmentMadeYear_2014',
                       'AppointmentYear_2014', 'AppointmentMadeDay_02', 'AppointmentMadeDay_03',
                       'AppointmentMadeDay_08', 'AppointmentMadeDay_09', 'AppointmentMadeDay_10',
                       'AppointmentMadeDay_11', 'AppointmentMadeDay_12', 'AppointmentMadeDay_19',
                       'AppointmentMadeDay_20', 'AppointmentMadeDay_22', 'AppointmentMadeDay_25',
                       'AppointmentMadeDay_28', 'AppointmentMadeDay_31', 'AppointmentMDay_01',
                       'AppointmentMDay_02', 'AppointmentMDay_04', 'AppointmentMDay_05',
                       'AppointmentMDay_07', 'AppointmentMDay_08', 'AppointmentMDay_09',
                       'AppointmentMDay_10', 'AppointmentMDay_12', 'AppointmentMDay_13',
                       'AppointmentMDay_15', 'AppointmentMDay_19', 'AppointmentMDay_20',
                       'AppointmentMDay_22', 'AppointmentMDay_26', 'AppointmentMDay_27',
                       'AppointmentMDay_29', 'AppointmentMDay_30', 'AppointmentMDay_31',
                       'AppointmentMadeDayofWeek_Fri', 'AppointmentMadeDayofWeek_Mon',
                       'AppointmentMadeDayofWeek_Sat', 'AppointmentMadeDayofWeek_Thurs',
                       'AppointmentMadeDayofWeek_Tues']
    
    X_train_d = pre_process(train)
    X_train_s = X_train_d.loc[:,colnames_select]
    
    ########## tuened models ###################
    from sklearn.calibration import CalibratedClassifierCV
    
    from sklearn.linear_model import LogisticRegression
    best_lr = LogisticRegression(solver = 'sag', C =0.03, n_jobs=-1, max_iter= 500)
        
    from sklearn.ensemble import RandomForestClassifier
    best_rf = RandomForestClassifier(n_jobs=-1, random_state=0, max_depth=15, 
                                 min_samples_leaf=1, min_samples_split = 10, n_estimators= 200, max_features= 22)
    best_rf_c = CalibratedClassifierCV(best_rf)

    from xgboost import XGBClassifier
    best_xgb = XGBClassifier(n_estimators = 100, max_depth =5, subsample =0.9, colsample_bytree = 1,
                             reg_lambda = 0.15, objective ='binary:logistic', n_jobs = -1)
    best_xgb_c = CalibratedClassifierCV(best_xgb)
    
    from sklearn.neural_network import MLPClassifier
    best_nn = MLPClassifier(hidden_layer_sizes =(10,), alpha = 0.07 )
    best_nn_c = CalibratedClassifierCV(best_nn)
    
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    best_svc = LinearSVC(C = 0.03 )
    best_svc_c = CalibratedClassifierCV(best_svc)
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    best_lda = LinearDiscriminantAnalysis(solver ='lsqr', shrinkage = 0.02)
    
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    best_qda = QuadraticDiscriminantAnalysis( reg_param = 1.8)
    
    
    from sklearn.ensemble import VotingClassifier
    eclf2 = VotingClassifier(estimators=[('rf', best_rf_c), ('xgb', best_xgb_c), 
                                         ('nn', best_nn)], voting='soft')
    eclf2.fit(X_train_s, X_label)
    
    ######### Prediction data preparation #################

    public = pd.read_csv("Predict_NoShow_PublicTest_WithoutLabels.csv", index_col="ID")
    public_prepro = pre_process(public)
    left, public_d = X_train_d.align(public_prepro, join='outer', axis=1, fill_value=0)
    public_s = public_d.loc[:,colnames_select]
    
    private = pd.read_csv("Predict_NoShow_PrivateTest_WithoutLabels.csv", index_col="ID")
    private_prepro = pre_process(private)
    left, private_d = X_train_d.align(private_prepro, join='outer', axis=1, fill_value=0)
    private_s = private_d.loc[:,colnames_select]
    
    
    
    def outputs(estimator = eclf2, predict_label = True):
    
        public_noshow_prob = estimator.predict_proba(public_s)[:,1].reshape(60000,1)
        public_id = public.index.values.reshape(60000,1)
        public_noshow_label = ((public_noshow_prob>0.48)*1).reshape(60000,1)
    
        private_noshow_prob = estimator.predict_proba(private_s)[:,1].reshape(60000,1)
        private_id = private.index.values.reshape(60000,1)
        private_noshow_label = ((private_prob>0.48)*1).reshape(60000,1)
        
        if predict_label == True:
            public_data = np.concatenate([public_id, public_noshow_prob, public_noshow_label], axis = 1 )
            private_data = np.concatenate([private_id, private_noshow_prob, private_noshow_label], axis = 1 )
        else:
            public_data = np.concatenate([public_id, public_noshow_prob], axis = 1 )
            private_data = np.concatenate([private_id, private_noshow_prob], axis = 1 )
        
        np.savetxt("public.csv", public_data, delimiter=",")
        np.savetxt("private.csv", private_data, delimiter=",")
        
        return None
    

        
    outputs()
    
    
    return None  
    
    
    
    

test = predictNoshow()