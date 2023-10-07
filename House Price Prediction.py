#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


Data = pd.read_csv('Housing.csv')


# In[20]:


Data


# In[21]:


Data.head()


# In[22]:


Data.info()


# In[23]:


Data.describe()


# In[16]:


sns.pairplot(Data)
plt.show()


# In[25]:


X = Data[['area', 'bedrooms', 'bathrooms','stories', 'parking']]

y = Data['price']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


lm = LinearRegression()


# In[30]:


lm.fit(X_train,y_train)


# In[31]:


LinearRegression()


# In[32]:


print(lm.intercept_)


# In[33]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[34]:


predictions = lm.predict(X_test)


# In[35]:


plt.scatter(y_test,predictions)
plt.show()


# In[36]:


sns.displot((y_test-predictions),bins=50,color="green")
plt.show()


# In[37]:


from sklearn import metrics


# In[38]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




