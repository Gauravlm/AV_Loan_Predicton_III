# Analytics vidya proect : Loan Prediction
# location of this csv file is in AV\LoaN_prediction|python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

#train = pd.read_csv('train.csv')
train = pd.read_csv('train.csv') # reading the data set using pandas

train.head()  # see top few rows

train.describe()  # summry of numerical variable

# check non-numeriacal variables 

train['Property_Area'].value_counts()

#ploting historgram of ApplicantIncome
train['ApplicantIncome'].hist(bins=50)

#ploting boxplot of ApplicantIncome for outliers
train.boxplot(column= 'ApplicantIncome')

# ApplicantIncome with Education
train.boxplot(column = 'ApplicantIncome', by= 'Education')
           
#ApplicantIncome with Gender
train.boxplot(column='ApplicantIncome',by='Gender')   

# histogram of loanamount
train['LoanAmount'].hist(bins=50) 

'''
LoanAmount has missing and well as extreme values values, 
while ApplicantIncome has a few extreme values,
 which demand deeper understanding. 
We will take this up in coming sections.
'''

#==============================================================================
 
 temp1 = train['Credit_History'].value_counts(ascending =True)
 
 temp2 =train.pivot_table(values='Loan_Status',index=['Credit_History'],
                          aggfunc=lambda x: x.map({'Y':1,'N':0}).mean()) # lamda use for save Y as 1 and N as 0
 
 print ('Frequency table for credit History')
 print (temp1)
 
 print ('\n Probability of geting loan for each credit history class')
 print(temp2)
 
 
 
#==============================================================================
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit History' )
ax1.set_ylabel('count of applicant')
ax1.set_title('Applicant by Credit History')
temp1.plot(kind = 'bar')

ax2= fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title('Probability of getting loan by Credit History ')

#############################################################################
temp3 = pd.crosstab(train['Credit_History'],train['Loan_Status'])
temp3.unstack().plot(kind='bar',stacked=True,color=['red','blue'],grid=False)
# temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)
# check missing values for each columns
train.apply(lambda x: sum(x.isnull()),axis=0)
'''
axis =1 or 0
0 or ‘index’: apply function to each column
1 or ‘columns’: apply function to each row
'''

# handling missing values form Gender, married,self_employed with there Mode
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace =True )

# boxplot loan amount, education Selp_employed
train.boxplot(column='LoanAmount',by =['Education','Self_Employed'])

# check Self_Employed missing values
train['Self_Employed'].value_counts()

# fill missing value as 'No' because 86% is 'No'
train['Self_Employed'].fillna('No',inplace =True)

# filling missing values in Education and Self_Employed using median
table = train.pivot_table(values = 'LoanAmount', columns = 'Education', index= 'Self_Employed',
                          aggfunc= np.median)

# define funtion to return values of pivote table
def fage(s):
    return(train.loc[s['Self_Employed'],x['Education']])

# Replace missing values
train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage,axis=1),inplace =True)

# treating LoanAmount outliers
train['LoanAmount_log']= np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
################################################################################
'''
One intuition can be that some applicants have lower income but strong support Co-applicants. 
So it might be a good idea to combine both incomes as total income and take a log transformation of the same.
'''

train['TotalIncome']= train['ApplicantIncome']+train['CoapplicantIncome']
train['TotalIncome_log']= np.log(train['TotalIncome'])
train['TotalIncome_log'].hist(bins=20)


###############################################################################

# Building predictive model in Python

from sklearn.preprocessing import LabelEncoder

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le =LabelEncoder()

for i in var_mod:
    train[i] = le.fit_transform(train[i])
train.dtypes

###############################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold  #  k-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# gernric function for making classification model and accesing performance 

def classification_model(model,data,predictors,outcome):
    #fit the model
    model.fit(data[predictors],data[outcome])
    
    # make prediction on training set
    predictions = model.predict(data[predictors])
    
    #print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    # perform k-fold cross validation with 5 fold
    kf= kFold(data.shape[0],n_fold=5)
    error=[]
    for train1,test in kf:
        #filter training data
        train_predictors= (data[predictors].iloc[train1,:])
        
        #target we are using to train the algoritham
        train_target = data[outcome].iloc[train1]
        
        #training algoithm using target and predictors
        model.fit(train_predictors,train_target)
        
        #record error from each cross-validation run 
        error.append(model.score(data[predictors].iloc[test,:],data[outcome].iloc[test]))
        
    print("Cross Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
    # Fit the model again so that it can be reffered outside the function
    model.fit(data[predictors],data[outcome])


