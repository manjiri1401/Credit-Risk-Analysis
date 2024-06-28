# Credit-Risk-Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from collections import Counter
import warnings
warnings.filterwarnings('ignore')       #deprications warnings will be ignore and there will be no warings

train_input = pd.read_csv("Credit_Risk_Train_data.csv")
validate_input = pd.read_csv("Credit_Risk_Validate_data.csv")

#this will shows all the column of the 2 data sets

print(train_input.columns)
print(validate_input.columns)

#The last column has a different name in both
#lets make the sname same and then merge them together
#so that wee can fill the missing values simulteneously
#inplace true means that it will be the permanent change in the data frame

validate_input.rename(columns={'outcome': 'Loan_Status'}, inplace = True)

#data_all is the data frame
#here we have concatinate means merged the 2 data frame together
#ingore_indx means that the original index will be removed
#shape means 981 rows and 13 columns

data_all = pd.concat([train_input,validate_input],ignore_index=True)
data_all.shape

# this will show how many rows are there in the data frame
#this is to check the correct data

data_all.tail()

plt.figure(figsize=(20,10))
sns.heatmap(data_all.isnull(), cbar=False)

#this gives the missing value of all the columns
#this will first convert into boolean and then all the null values
#means in Gender there are total 24 null values which are NaN

data_all.isnull().sum()

#this shows what is the status of the Gender column
#according to Mode method change the values of those who have higher values in the data


Counter(data_all['Gender'])
#this will fill the NaN value of all the Male in the perrmanent data frame

data_all.fillna({'Gender': 'Male'}, inplace = True)

#check if filled

Counter(data_all['Gender'])

#Lets fill Married now

print(Counter(data_all['Married']))

#this shows that most of them are married

data_all.fillna({'Married': 'Yes'}, inplace=True)

#isnull is for checking the NaN values

data_all.isnull().sum()

Counter(data_all['Dependents'])

#Lests see the Dependents with respect to Marriage
#this shows how many are there who are married plus have the dependency
#True means no value and False means they have some values

pd.crosstab(data_all['Married'],data_all['Dependents'].isnull())

#this guve the data which are married and have the dependency on them

pd.crosstab(data_all['Dependents'],data_all['Married'])

# for the bachelors, lets fill the missing dependents as 0
# lests find the index of all rows with Dependents missing ans married NO
#tolist means that the data frae is onverted into list

bachelor_nulldependent = data_all[(data_all['Married']=='No') &
                                 (data_all['Dependents'].isnull())].index.tolist()
print(bachelor_nulldependent)

#this means making the list into 0

data_all['Dependents'].iloc[bachelor_nulldependent]='0'

# checking the number of NaN values

Counter(data_all['Dependents'])

#for remaning 16 missing dependents
#let see how many dependents male and Female have

pd.crosstab(data_all['Gender'],data_all['Dependents'])

#lets see the gender of the 16 missing dependents

pd.crosstab(data_all['Gender'],data_all['Dependents'].isnull())

#this will give the Male who are married and have dependencies on them

pd.crosstab((data_all['Gender']=='Male')&
           (data_all['Married']=='Yes'), data_all['Dependents'])

#lets fill the dependent with 1

data_all['Dependents'].iloc[data_all[data_all['Dependents'].isnull()].index.tolist()] ='1'

Counter(data_all['Self_Employed'])
data_all.fillna({'Self_Employed': 'No'}, inplace = True)
data_all.isnull().sum()

# To check if any rows with both LoanAmount and Loan_Amount_Term as NaN

pd.crosstab(data_all['LoanAmount'].isnull(),
           data_all['Loan_Amount_Term'].isnull())
# Cross tab is for corealtion of 2 aor more values

pd.crosstab(data_all['LoanAmount'].isnull(), data_all['Loan_Amount_Term'])

# Have taking the mean of every term value to get the avg

data_all.groupby(data_all['Loan_Amount_Term'])['LoanAmount'].mean()

# Lets fill the missing values in LoanAmount
#with the mean of the respective Loan_Term
# we see that 180 and 240 has the almost same Loan amount 128-131
# and 360 has high i.e 144
# so lets fill only 360 by 144
# all remaining by 130

data_all['LoanAmount'][(data_all['LoanAmount'].isnull())
                     & (data_all['Loan_Amount_Term']==360)] == 144    #means in 360 we are replacing NaN value with 144

data_all['LoanAmount'][(data_all['LoanAmount'].isnull())
                     & (data_all['Loan_Amount_Term']==480)] == 137    #means in 480 we are replacing NaN value with 137

# here we have taking the comman value as 130 and fill the NaN of remaning term values

data_all['LoanAmount'][(data_all['LoanAmount'].isnull())]=130
# Lets fill Loan Amount Term

(data_all['Loan_Amount_Term']).value_counts()

# lets fill the laon Tenure by the mode i.e 360

data_all['Loan_Amount_Term'][data_all['Loan_Amount_Term'].isnull()]=360

data_all.isnull().sum()
data_all['Credit_History'].value_counts()

 # here we are taking the correlation of the credit history with every parameters

data_all.corr()
# now wee will check the relation of credit history with each parameters

pd.crosstab(data_all['Gender'], data_all['Credit_History'])

# Gender makes no difference
pd.crosstab(data_all['Self_Employed'], data_all['Credit_History'])
pd.crosstab(data_all['Education'], data_all['Credit_History'])
pd.crosstab(data_all['Married'], data_all['Credit_History'])
# We have filled NaN value of Credit History as 1

data_all.fillna({'Credit_History':1}, inplace = True)
data_all.isnull().sum()

# This means that firstly we have to make every parameters 0, as in ML NaN is not allowed

data_all.head()
data_all['Dependents'].value_counts()
data_all['Dependents'][data_all['Dependents']=='3+']='3';
data_all['Dependents'].head()
data_all['Dependents'] = data_all['Dependents'].astype(int)
data_all['Dependents'].head()
data_all_new = pd.get_dummies(data_all.drop(['Loan_ID'],axis = 1),
                             drop_first=True)
data_all_new.head()
X = data_all_new.drop(['Loan_Status_Y'], axis = 1)
y = data_all_new['Loan_Status_Y']
X.head()
y.head()
# Data Spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
X_train.shape
X_test.shape

# Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the  training data

scaler.fit(X)# Now apply the transformations to the data:

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train[:5]

# Traning the model using K-nn

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
parameters = {'n_neighbors' : [3,5,11,19],
             'weights' : ['uniform','distance'],
             'metric' : ['minkowski', 'manhattan']}
clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=3,
                  verbose=1,n_jobs=-1)
clf.fit(X_train,y_train)
clf.best_score_
clf.best_params_
clf = KNeighborsClassifier(metric = 'minkowski' , n_neighbors = 5, weights = 'uniform')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix , accuracy_score, classification_report
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

# END
