# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:04:17 2020

@author: Bharanitharan
"""
#######################################################################################################################################

                                                # part 2 try
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

#setting path
os.chdir('E:\\Assignement')

#importing data

data = pd.read_csv('bank-full.csv',delimiter = ';')
data.describe()

#audit the data
data.head()
print(data.shape)

data.dtypes

data.isnull().sum()

# see distinct values in the column

for col in data.columns:
    print(data[col].value_counts())

# missing value in JOB, contact , potcome , educaion

for col in data.columns:
    missing = (data[col]=='unknown').sum()
    print(col,':',missing)
 
# checking correlatiin to fill the missing values
# filling unknown with mode of job
data['job'].value_counts()
data['job'] = data['job'].replace(['unknown'],data['job'].mode()[0])
data['job'].mode()[0]

# filling unknown with the mode of education
data['education'].value_counts()
data['education'] = data['education'].replace(['unknown'],data['education'].mode()[0])
data['education'].mode()[0]

data1 = data
#creating another feature named Age Group
category = pd.cut(data1.age,bins=[13,19,29,59,99],labels=['Teen Aged','Youth','Middle Aged','Old Aged'])
data1.insert(1,'Age Group',category)

data1 = data1.drop(['age'],axis = 1)
data1.head()

value={'may' : 'Busy Month','jun':'Busy Month','jul':'Busy Month','aug':'Busy Month',
     'feb': 'Moderate Month','jan': 'Moderate Month','apr': 'Moderate Month','nov' : 'Moderate Month',
     'mar': 'Low Month','sep': 'Low Month','oct': 'Low Month','dec' : 'Low Month'}
category=data1['month'].map(value)
data1.insert(10,'Weighted Months',category)

data1 = data1.drop(['month'],axis = 1)
data1.head()


day_flag = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thrusday', 6: 'Friday', 7: 'Saturday', 8: 'Sunday', 9: 'Monday',10:'Tuesday', 11: 'Wednesday', 12 :'Thrusday',13: 'Friday', 14:'Saturday', 15:'Sunday', 16: 'Monday', 17: 'Tuesday', 18: 'Wednesday', 19:'Thrusday', 20: 'Friday', 21: 'Saturday', 22:'Sunday', 23: 'Monday', 24: 'Tuesday', 25:'Wednesday', 26:'Thrusday', 27:'Friday',28:'Saturday', 29: 'Sunday', 30:'Monday', 31:'Tuesday'}
data1['day'].replace(day_flag, inplace = True)


transformed_data  = data1 

transformed_data['balance'].isna().sum()

# univariate analysis shows that there is more skew in balance duration and pdays and previous

# droping skewnesss using log

transformed_data = transformed_data.drop(['previous'],axis = 1)
transformed_data['previous'] = np.log(cleaned_data['previous']).replace([np.inf, -np.inf], 0)
transformed_data['previous'].skew()

#transformed_data = transformed_data.drop(['balance'],axis = 1)
#transformed_data['balance'] = np.log(cleaned_data['balance']).replace([np.inf, -np.inf], 0)
#transformed_data['balance'].skew()
#transformed_data['balance'].isna().sum()

transformed_data = transformed_data.drop(['duration'],axis = 1)
transformed_data['duration'] = np.log(cleaned_data['duration']).replace([np.inf, -np.inf], 0)
transformed_data['duration'].skew()

transformed_data = transformed_data.drop(['campaign'],axis = 1)
transformed_data['campaign'] = np.log(cleaned_data['campaign']).replace([np.inf, -np.inf], 0)
transformed_data['campaign'].skew()


for i in numeric_col:
    print (i,':',transformed_data[i].skew())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def categorical_variable(dataframe):
    variable_name=[i for i in dataframe.columns if dataframe.dtypes[i]=='object']
    for x in variable_name:
        dataframe[x]=le.fit_transform(dataframe[x])
    return dataframe

categorical_variable(transformed_data)
transformed_data.columns
transformed_data['Age Group'] = le.fit_transform(transformed_data['Age Group'])
transformed_data.dtypes
# feature selecition
from sklearn.model_selection import train_test_split 

y = transformed_data['y']
x = transformed_data
x = x.drop(['y'],axis= 1)

x.shape

# spliting data base into 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)

# logistic regression
import statsmodels.api as sn
logit = sn.Logit(y_train,x_train)
result = logit.fit()
result.summary()
preds = result.predict(x_train)
preds= np.where(preds>0.5,1,0)

### calculating f1score
from sklearn.metrics import f1_score
f1_sc_logistic_train = f1_score(y_train,preds)
from sklearn.metrics import confusion_matrix
cof_mat = confusion_matrix(y_train,preds)
from sklearn.metrics import accuracy_score
arcy_sc_logistic_train = accuracy_score(y_train,preds)
preds_ytest = result.predict(x_test)
preds_ytest = np.where(preds_ytest>0.5,1,0)
f1_sc_logistic_test  = f1_score(y_test,preds_ytest)
arcy_sc_logistic_test = accuracy_score(y_test,preds_ytest)


##### decision treee
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(x_train,y_train)
pred_dtc = DTC.predict(x_train)
f1_sc_decisison_train = f1_score(y_train,pred_dtc)
arcy_sc_decisison_train = accuracy_score(y_train,pred_dtc)
cof_mat = confusion_matrix(y_train,pred_dtc)
preds_dc_ytest = DTC.predict(x_test)
f1_sc_decisison_test = f1_score(y_test,preds_dc_ytest)
arcy_sc_decisison_test = accuracy_score(y_test,preds_dc_ytest)



##  supoort vector machine
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train,y_train)
preds_svm = svm.predict(x_train)
cm_svm = confusion_matrix(y_train,preds_svm)
acq_score_svm_train = accuracy_score(y_train,preds_svm)
f1score_svm_train = f1_score(y_train,preds_svm)
preds_svm_train = svm.predict(x_test)
f1score_svm_test = f1_score(y_test,preds_svm_train)
cm_svm = confusion_matrix(y_test,preds_svm_train)
arcy_sc_svm_test = accuracy_score(y_test,preds_svm_train)


# sk learns nural network
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(10,10,10), verbose = True, solver='sgd', max_iter=300)
MLP.fit(x_train,y_train)
pred_mlp = MLP.predict(x_train)
cof_mat = confusion_matrix(y_train,pred_mlp)
acq_score_nural = accuracy_score(y_train,pred_mlp)
acq_score_nural
f1score_neural_train = f1_score(y_train,pred_mlp)
pred_ytest_MLP = MLP.predict(x_test)
f1score_neural_test = f1_score(y_test,pred_ytest_MLP)
acq_for_TEST = accuracy_score(pred_ytest_MLP,y_test)
acq_for_TEST


# random forest
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)
pred_rfc = RFC.predict(x_train)
cof_rfc_train = confusion_matrix(y_train,pred_rfc)
acq_score_rfc_train = accuracy_score(y_train,pred_rfc)
f1score_rfc_train = f1_score(y_train,pred_mlp)
pred_ytest_rfc = RFC.predict(x_test)
f1score_rfc_test = f1_score(y_test,pred_ytest_rfc)
acq_score_rfc_test = accuracy_score(y_test,pred_ytest_rfc)


# comparing accuracy f1 score of models
print("logistic ","accuracy in train = ",arcy_sc_logistic_train)
print("logistic ","accyracy for test = ",arcy_sc_logistic_test)
print("logistic ","f1score of train =  ",f1_sc_logistic_train)
print("logistic ","f1score of test =  ",f1_sc_logistic_test)
print("decision tree ","accuracy in train = ",arcy_sc_decisison_train)
print("decision tree  ","accyracy for test = ",arcy_sc_decisison_test)
print("decision tree  ","f1score of train =  ",f1_sc_decisison_train)
print("decision tree  ","f1score of test =  ",f1_sc_decisison_test)
print("SVM ","accuracy in train = ",acq_score_svm_train)
print("SVM ","accyracy for test = ",arcy_sc_svm_test)
print("SVM ","f1score of train =  ",f1score_svm_train)
print("SVM ","f1score of test =  ",f1score_svm_test)
print("nerual network ","accuracy in train = ",acq_score_nural)
print("nerual network ","accyracy for test = ",acq_for_TEST)
print("nerual network ","f1score of train =  ",f1score_neural_train)
print("nerual network ","f1score of test =  ",f1score_neural_test)
print("randomforest ","accuracy in train = ",acq_score_rfc_train)
print("randomforest ","accyracy for test = ",acq_score_rfc_test)
print("randomforest ","f1score of train =  ",f1score_rfc_train)
print("randomforest ","f1score of test =  ",f1score_rfc_test)

