#!/usr/bin/env python
# coding: utf-8

# # Task 1:  Prediction Using Supervised ML - GRIP - Data Science And Business Analytics - JAN 2021

# # Author: Karan Shah

 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 %matplotlib inline

# In[3]:


data = pd.read_csv('http://bit.ly/w-data')
print(data)


# In[4]:


data.describe()


# In[5]:


data.head()


# In[7]:


data.plot(kind='scatter',x='Hours',y='Scores')
plt.title('Hours studied vs Scores obtained')
plt.xlabel('Hours studied')
plt.ylabel('Scores obtained')
plt.show()


# In[8]:


x=data.iloc[:,:-1]
y=data.iloc[:,1]


# In[9]:


x


# In[10]:


y


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[12]:


x_train


# In[13]:


x_test


# In[14]:


y_train


# In[15]:


y_test


# In[16]:


len (x_train),len(x_test),len(y_train),len(y_test)


# In[17]:


#training the model


# In[18]:


from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x_train,y_train)
print('completed')


# In[26]:


**Plotting the regression line**


# In[22]:


line=Lr.coef_*x+Lr.intercept_
#Plotting for testing data
plt.scatter(x,y)
plt.plot(x,line);
plt.show()
print(("Equation of line y : "), str(Lr.coef_[0]),('x+'),str(Lr.intercept_))

**Testing The Model**
# In[25]:


print('Training Score : ', round(Lr.score(x_train,y_train)*100,2),'%')
print('Testing Score : ', round(Lr.score(x_test,y_test)*100,2),'%')


# In[27]:


y_pred = Lr.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  


# In[28]:


df


# In[29]:


from sklearn.metrics import r2_score
from math import sqrt
from sklearn import metrics


# In[30]:


k = x_test.shape[1]
n = len(x_test)
r2 = r2_score(y_test,y_pred)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)


# In[31]:


print('R2 = ',r2,'\nAdjusted R2 = ',adj_r2)


# In[32]:


print('Mean Absolute error : ',metrics.mean_absolute_error(y_test,y_pred))


# In[33]:


Hours = 9.25
pred = Lr.predict([[Hours]])
print('Number of Hours : {}'.format(Hours))
print('Predicted Score : {}'.format(pred[0]))


# In[ ]:





# In[ ]:




