#Importing necessary library 
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

# for contact and poutcome 
# contact and poutcome has missig value nearly 30% percent so droping these variables

cleaned_data = data.drop(['contact','poutcome'],axis = 1)

# checking missing values in new data set 
for col in cleaned_data.columns:
    missing = (data[col]=='unknown').sum()
    print(col,':',missing)


# univatiate analysis
# plotting data set to show how its distributed
plt.hist(cleaned_data['age'])  

cleaned_data['age'].skew()
numeric = cleaned_data.dtypes != object
numeric_col = cleaned_data.columns[numeric].tolist()

for i in numeric_col:
    print (i,':',cleaned_data[i].skew())

for i in numeric_col:
    plt.boxplot(cleaned_data[i])
    plt.title(i)
    plt.show()
transformed_data  = cleaned_data 

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

# Bivariata analysis

## catagorical vaiable
import researchpy as rp
category=['job','education','marital','default','housing','loan','contact','month','poutcome']
for c in category :
    table, results = rp.crosstab(data[c],data['y'], test= 'chi-square')
    print(results)
    print('='*45)

# croosstab with catagorical
for c in category :
    table = pd.crosstab(data[c],data['y'])
    table.plot(kind='bar')

# for numeric variable
corelation = data.corr()
ax=plt.subplots(figsize=(9,7))
sns.heatmap(corelation,annot = True)


### multivaite analysis

sns.pairplot(data,hue='y',palette='coolwarm')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def categorical_variable(dataframe):
    variable_name=[i for i in dataframe.columns if dataframe.dtypes[i]=='object']
    for x in variable_name:
        dataframe[x]=le.fit_transform(dataframe[x])
    return dataframe

categorical_variable(transformed_data)
transformed_data.columns

# feature selecition
from sklearn.model_selection import train_test_split 

y = transformed_data['y']
x = transformed_data
x = x.drop(['y'],axis= 1)

x.shape

# spliting data base into 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)



#xgboost
import xgboost as xgb
xgb_class = xgb.XGBClassifier(n_estimators = 20)
xgb_class.fit(x_train,y_train)
preds_class = xgb_class.predict(x_train)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
f1train_xgboost = f1_score(y_train,preds_class)
f1train_xgboost
cm = confusion_matrix(preds_class, y_train)
acc_score = accuracy_score(preds_class, y_train)
print(cm)
print(acc_score)
predic = xgb_class.predict(x_test)
f1test_xgboost = f1_score(predic, y_test)
f1test_xgboost
cm = confusion_matrix(predic, y_test)
acc_score = accuracy_score(predic, y_test)
print(cm)
print(acc_score)




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
f1score_rfc_train = f1_score(y_train,)
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
#print("SVM ","accuracy in train = ",acq_score_svm_train)
#print("SVM ","accyracy for test = ",arcy_sc_svm_test)
#print("SVM ","f1score of train =  ",f1score_svm_train)
#print("SVM ","f1score of test =  ",f1score_svm_test)
print("nerual network ","accuracy in train = ",acq_score_nural)
print("nerual network ","accyracy for test = ",acq_for_TEST)
print("nerual network ","f1score of train =  ",f1score_neural_train)
print("nerual network ","f1score of test =  ",f1score_neural_test)
print("randomforest ","accuracy in train = ",acq_score_rfc_train)
print("randomforest ","accyracy for test = ",acq_score_rfc_test)
print("randomforest ","f1score of train =  ",f1score_rfc_train)
print("randomforest ","f1score of test =  ",f1score_rfc_test)












plt.boxplot(np.log(cleaned_data['previous']).replace([np.inf, -np.inf], 0))
(cleaned_data['previous'] == 0).sum()
plt.plot(cleaned_data['previous'])
cleaned_data['previous'].max()






chi = chi2_contingency(table)
print(chi)
corelation = data.corr()
ax=plt.subplots(figsize=(20,20))
sns.heatmap(correl,vmax=1,square=True,annot = True)







# handeling imbalnced data set

from imblearn.over_sampling import SMOTE        
sm = SMOTE(random_state =27)
x_train,y_train = sm.fit_sample(x_train,y_train)









