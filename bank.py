
# coding: utf-8

# In[124]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# In[125]:


bank = pd.read_csv("/Users/durai/Desktop/ML/datasets/bank/bank.csv",delimiter=";" )


# In[126]:


print(bank.groupby("y").size())


# In[127]:


bank.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()


# In[128]:


bank.hist()
plt.show()


# In[129]:


scatter_matrix(bank)
plt.show()


# In[144]:


X= bank.drop('y', axis=1)
Y = bank['y']
le = LabelEncoder()
obj_x = X
obj_x["job_code"] = le.fit_transform(obj_x["job"])
obj_x["marital_code"] = le.fit_transform(obj_x["marital"])
obj_x["education_code"] = le.fit_transform(obj_x["education"])
obj_x["default_code"] = le.fit_transform(obj_x["default"])
obj_x["housing_code"] = le.fit_transform(obj_x["housing"])
obj_x["loan_code"] = le.fit_transform(obj_x["loan"])
obj_x["poutcome_code"] = le.fit_transform(obj_x["poutcome"])

obj_xt= obj_x[["age","job_code","marital_code","education_code","default_code","housing_code","loan_code",
               "poutcome_code","day"]]
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(obj_xt, Y, test_size=0.35, random_state=42)


# In[146]:


X_train.head()


# In[147]:



# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# In[148]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[ ]:





# In[ ]:





# In[149]:


results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=7)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[150]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[151]:


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




