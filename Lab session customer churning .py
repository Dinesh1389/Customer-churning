#!/usr/bin/env python
# coding: utf-8

# # PROJECT OBJECTIVE: Predict the customer churn 

# In[1]:


## Importing the required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Lets import the data

data_df=pd.read_csv('WA_Fn_UseC_Telco-Customer-Churn.csv')


# In[3]:


##Checking to see if the data has been imported successfully

data_df.head()


# In[4]:


## Checking the Shape of the Data

data_df.shape

## There are 7043 rows and 21 columns in the dataset


# In[5]:


## DTYPES of the Columns

data_df.dtypes

## We can see that there are many data types as objects which we will need to convert to numeric during further building


# In[6]:


## Changing the Total Charges column from Object to Numeric as it has numeric data

data_df['TotalCharges']=pd.to_numeric(data_df['TotalCharges'],errors='coerce')

## Calling the dtype function to confirm the change of the datatype for Total Charges

data_df['TotalCharges'].dtypes

## The Datatype has been converted to numeric


# In[7]:


## Looking at the unique values for all the columns

for i in data_df.columns:
    print(f" Unique{i}'s count: {data_df[i].nunique()}")
    print(f" {data_df[i].unique()}\n")


# In[8]:


## Checking for missing Values

data_df.isnull().sum()

## We have a total of 11 null values in the Total Charges column


# In[9]:


## DESCRIPTIVE ANALYSIS

## The reason for less columns showing in describe is because we have many columns with the DataType as Object which need to converted to Numeric


# In[10]:


data_df.describe().T

INSIGHTS

Average Tenure is 32 months 25% of customers have a tenure of 9 months

# In[11]:


## Duplicate Check


# In[12]:


data_df.duplicated().sum()


# In[13]:


## There are no duplicates in the data


# In[14]:


## The data is imbalanced data and needs to be balanced which will be done during featue selection and engineering


# In[15]:


data_df['Churn'].value_counts()


# In[16]:


## Deleting the Redundant column Customer ID as we cant make any changes on the same


# In[17]:


data_df=data_df.drop(['customerID'],axis=1)


# In[18]:


data_df.head()


# In[19]:


## We need to take care of the Output i.e. Churn. We need to map the values and cast them to the Columns

## We can see the values are now in 0's and 1's


# In[20]:


churn_map = {'No': 0, 'Yes': 1}
data_df['Churn'] = data_df['Churn'].map(churn_map)
data_df['Churn']


# In[21]:


data_df['Churn'].isnull().sum()


# In[22]:


data_df.info()


# In[23]:


## Using One Hot Encoding to convert all the Object Datatypes to Numeric


# In[24]:


data_df_dummy=pd.get_dummies(data_df)


# In[25]:


data_df_dummy.head()


# In[26]:


## Group 1--Fins out the customer info with Churn


# In[27]:


data_df_dummy.dtypes


# In[28]:


data_df.head()


# In[29]:


## Customer Information with respect to Churn. 


# In[30]:


C1=data_df[['gender','SeniorCitizen','Partner','Dependents']]

for i in C1:
    plt.figure(i)
    sns.countplot(data=data_df,x=i,hue='Churn',palette='rainbow')


# In[31]:


data_df.info()


# In[32]:


## CUSTOMER INFORMATION

gender=data_df[data_df['Churn']==1]['gender'].value_counts()
gender=[gender[0]/sum(gender)*100,gender[1]/sum(gender)*100]

Seniorcitizen=data_df[data_df['Churn']==1]['SeniorCitizen'].value_counts()
Seniorcitizen=[Seniorcitizen[0]/sum(Seniorcitizen)*100,Seniorcitizen[1]/sum(Seniorcitizen)*100]

Partner=data_df[data_df['Churn']==1]['Partner'].value_counts()
Partner=[Partner[0]/sum(Partner)*100,Partner[1]/sum(Partner)*100]

Dependents=data_df[data_df['Churn']==1]['Dependents'].value_counts()
Dependents=[Dependents[0]/sum(Dependents)*100,Dependents[1]/sum(Dependents)*100]


# In[33]:


print(gender)
print(Seniorcitizen)
print(Partner)
print(Dependents)


# In[34]:


plt.figure(figsize=(15,10),dpi=200)
sns.heatmap(data_df.corr(),annot=True,cmap='coolwarm',fmt='.0%')


# In[35]:


plt.figure(figsize=(6,5))
sns.heatmap(data_df.corr(), annot = True, cmap = "Greens");


# In[36]:


data_df.head()


# In[37]:


#We see that the columns are of uint8 type and we need them as floats or integers


# In[38]:


for column_name in data_df_dummy.columns:
    data_df_dummy[column_name] = data_df_dummy[column_name].astype(float)
data_df_dummy.info()


# In[39]:


data_df_dummy.head()


# In[42]:


#We will make a barplot of the correlated features


# In[43]:


values = data_df_dummy.corr()['Churn'].sort_values(ascending=False)[1:].values
index = data_df_dummy.corr()['Churn'].sort_values(ascending=False)[1:].index


# In[44]:


#Now we will create the plot


# In[97]:


palette = 'Set1'


# In[98]:


fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
sns.barplot(x=index, y=values, palette=palette,ax=ax)
plt.xticks(rotation=90)
plt.title('Correlation Graph of the features with Churning')
plt.ylabel('Correlation', fontsize=15)
plt.xlabel('Features', fontsize=15)
plt.show()


INSIGHTS FROM THE BAR PLOT

1. The Longer the Tenure of the customer the less he is likely to Churn
2. The customers with Two year contract are less likely to Churn
3. More utilities that the customers get when offline the less there are chances of them to Churn
4. Another interesting point is that if the total charges are less then the chances of customers chruning are 
5. The elder the person more likely they will chrun
6. As the monthly charges go up the customer is likely to churn
7. payment method is also having an impact on the customer chrun
8. The Highest factor of customer chruning is Fiber Optic
# In[47]:


## Lets check the number of Senior Citizens in the dataset

elderly_count=data_df_dummy['SeniorCitizen'].value_counts()


# In[48]:


#Now we will create the figure
colors = sns.color_palette('Set1')
explode = [0.02, 0.035]
labels = elderly_count.unique()

plt.figure(figsize=(5, 4), dpi=150, facecolor='white')
plt.pie(elderly_count, labels=['Not Senior Citizen', 'Senior Citizen'], colors=colors, 
       explode=explode, autopct='%1.1f%%')
plt.title('Piechart of Senior Citizens', fontweight='bold')
plt.tight_layout()
plt.show()

USEFUL INSIGHTS

1. % Senior Citizens - There are only 16% of the customers who are senior citizens. 

Thus most of our customers in the data are younger people.
# ## Let's now explore the churn rate for all the customers

# In[49]:


churning_clients = data_df_dummy['Churn'].value_counts()

#making the plot

sns.set_palette('Set2')

plt.figure(figsize=(6, 6), dpi=100)
ax = sns.barplot(x=churning_clients.index.map({1:'Churn', 0:'No Churn'}), y=churning_clients.values)
plt.title('Countplot of the Customer Churn', fontweight='bold')
plt.grid(axis='y', linewidth=1, alpha=0.3)
plt.ylabel('Number of Customers')
plt.xlabel('Churn')

plt.tight_layout()
plt.show()

#Some Useful Insights

In general we can see that we do have high customer churn which the company needs to take care of

How ever the good news is that most of the customer are loyal and will not churn so the company needs to keep the same
# In[50]:


## Checking the Correlation between the Monthly Charges and Total Charges

plt.figure(figsize=(6, 6), dpi=100)
sns.scatterplot(data=data_df_dummy, x='MonthlyCharges', y='TotalCharges', alpha=0.6, marker='h')
plt.title('Scatterplot of the relation between charges', fontweight='bold')
plt.tight_layout()
plt.show()


# #Useful Insight
# We can see that there is a postiive correlation. If the Monthly charges are more the total charges will also be more

# In[51]:


## Calculating the Numerical Features so that we can peform the Outlier check

numerical_feature=[feature for feature in data_df_dummy.columns if data_df_dummy[feature].dtypes!='O']
print('Number of Numerical Features:', len(numerical_feature))

## Outlier Detection

for feature in numerical_feature:
    data_df_dummy[feature]=np.log(data_df_dummy[feature])
    data_df_dummy.boxplot(column=feature)
    plt.xlabel('feature')
    plt.ylabel('feature')
    plt.title('feature')
    plt.show()# INSIGHT: There are no ouliers in the dataset 
# In[52]:


## using the Mean to impute the missing value in total Charges

data_df_dummy['TotalCharges']=data_df_dummy['TotalCharges'].fillna(data_df_dummy['TotalCharges'].mean())


# In[53]:


data_df_dummy.head()


# In[54]:


## Splitting the data into X_train ,y_train using Train test Split

X=data_df_dummy.drop(['Churn'],axis=1)
y=data_df_dummy['Churn']


# In[55]:


X.info()


# In[56]:


X.head()

for column_name in X_train.columns:
    data_df_dummy[column_name] = data_df_dummy[column_name].astype(int)
data_df_dummy.info()
# In[57]:


from sklearn. model_selection import train_test_split


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[59]:


## Standardization using Standard Scaler

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[60]:


scaled_X_train=sc.fit_transform(X_train)
scaled_X_test=sc.fit_transform(X_test)


# In[61]:


scaled_X_test.shape


# MODEL BUILDING AFTER USING STANDARDIZATION

# In[62]:


## Importing the Logistic Regression Algorithm as this is a classification problem

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression() 


# In[63]:


## Fitting the X_train and Y_train on the model

model1=lr.fit(scaled_X_train,y_train)


# In[64]:


y_pred=model1.predict(scaled_X_test)


# In[65]:


from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


# In[66]:


print(classification_report(y_test,y_pred))


# In[67]:


cm = confusion_matrix(y_test,y_pred)
display = ConfusionMatrixDisplay(cm)
display.plot()
plt.show()


# In[68]:


## Running modal with RandomForest classification problem


# In[69]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[70]:


## Fitting the X_train and Y_train on the model

model2=rf.fit(scaled_X_train,y_train)


# In[71]:


y_pred_mod2=model2.predict(scaled_X_test)


# In[72]:


print(classification_report(y_test,y_pred_mod2))


# In[101]:


cm1 = confusion_matrix(y_test,y_pred_mod2)
display = ConfusionMatrixDisplay(cm1)
display.plot()
plt.show()


# In[74]:


## Running the Model with Decision Tree  classification problem

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[75]:


## Fitting the X_train and Y_train on the model

model3=dt.fit(scaled_X_train,y_train)


# In[76]:


y_pred_mod3=dt.predict(scaled_X_test)


# In[77]:


print(classification_report(y_test,y_pred_mod3))


# In[102]:


cm2 = confusion_matrix(y_test,y_pred_mod3)
display = ConfusionMatrixDisplay(cm2)
display.plot()
plt.show()


# In[79]:


## KNN Classification pproblem

from sklearn.neighbors import KNeighborsClassifier 


# In[80]:


model4=KNeighborsClassifier(n_neighbors=14)


# In[81]:


model4.fit(scaled_X_train,y_train)


# In[82]:


y_pred_knn=model4.predict(scaled_X_test)


# In[83]:


print(classification_report(y_test,y_pred_knn))


# In[84]:


cm4=confusion_matrix(y_pred_knn,y_pred)


# In[103]:


cm4


# In[86]:


#test_error_rates


# In[87]:


from sklearn.metrics import accuracy_score


# In[88]:


test_error_rates=[]
for k in range(1,30):
    model5=KNeighborsClassifier(n_neighbors=k)
    model5.fit(scaled_X_train,y_train)
    y_pred_test=model5.predict(scaled_X_test)
    test_error=1-accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)


# In[89]:


plt.plot(range(1,30),test_error_rates)
plt.ylabel('ERROR RATE')
plt.xlabel('K Neighbors')


# In[90]:


## Running ROC Curve for all models.


# In[91]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[92]:


FPR,TPR,_=roc_curve(y_test,y_pred)
plt.plot(FPR,TPR)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Logistic Regression')
plt.show()

FPR,TPR,_=roc_curve(y_test,y_pred_mod2)
plt.plot(FPR,TPR)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Random Forest Classifier')
plt.show()

FPR,TPR,_=roc_curve(y_test,y_pred_mod3)
plt.plot(FPR,TPR)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Decision Tree')
plt.show()

FPR,TPR,_=roc_curve(y_test,y_pred_knn)
plt.plot(FPR,TPR)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('K Nearest Neighbor')
plt.show()


# In[ ]:




