#!/usr/bin/env python
# coding: utf-8

# # Flight Price Prediction (EDA + Feature Engineering)

# In[153]:


# Importing basics libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[154]:


train_df=pd.read_excel('Data_Train.xlsx')
train_df.head()


# In[155]:


test_df=pd.read_excel('Test_set.xlsx')
test_df.head()


# In[156]:


final_df = pd.concat([train_df, test_df], ignore_index=True)


# In[157]:


final_df.head()


# In[158]:


final_df.tail()


# In[159]:


final_df.info()


# In[160]:


final_df['Date_of_Journey'].str.split('/').str[0]


# In[161]:


# Feature Engineering Process


# In[162]:


final_df['Date']=final_df['Date_of_Journey'].str.split('/').str[0]
final_df['Month']=final_df['Date_of_Journey'].str.split('/').str[1]
final_df['Year']=final_df['Date_of_Journey'].str.split('/').str[2]


# In[163]:


final_df.head(2)


# In[164]:


df["Date"]=df['Date_of_Journey'].apply(lambda x:x.split("/")[0])
df["Month"]=df['Date_of_Journey'].apply(lambda x:x.split("/")[1])
df["Year"]=df['Date_of_Journey'].apply(lambda x:x.split("/")[2])


# In[165]:


final_df['Date']=final_df['Date'].astype(int)
final_df['Month']=final_df['Month'].astype(int)
final_df['Year']=final_df['Year'].astype(int)


# In[166]:


final_df.info()


# In[167]:


final_df.drop('Date_of_Journey',axis=1,inplace=True)


# In[168]:


final_df.head(10)


# In[169]:


final_df['Arrival_Time'].str.split(' ').str[0]


# In[170]:


final_df['Arrival_Time']=final_df['Arrival_Time'].apply(lambda x : x.split(' ')[0])


# In[171]:


final_df['Arrival_hour']=final_df['Arrival_Time'].str.split(':').str[0]
final_df['Arrival_min']=final_df['Arrival_Time'].str.split(':').str[1]


# In[172]:


final_df.head(1)


# In[173]:


final_df['Arrival_hour']=final_df['Arrival_hour'].astype(int)
final_df['Arrival_min']=final_df['Arrival_min'].astype(int)


# In[174]:


final_df.drop('Arrival_Time',axis=1, inplace= True)


# In[175]:


final_df.head(5)


# In[176]:


final_df['Dept_hour']=final_df['Dep_Time'].str.split(':').str[0]
final_df['Dept_min']=final_df['Dep_Time'].str.split(':').str[1]
final_df['Dept_hour']=final_df['Dept_hour'].astype(int)
final_df['Dept_min']=final_df['Dept_min'].astype(int)
final_df.drop('Dep_Time',axis=1,inplace =True)


# In[177]:


final_df.info()


# In[178]:


final_df['Total_Stops'].unique()


# In[179]:


final_df['Total_Stops']=final_df['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4,'nan':1})


# In[180]:


final_df[final_df['Total_Stops'].isnull()]


# In[181]:


final_df.drop('Route',axis=1, inplace= True)


# In[182]:


final_df.head()


# In[183]:


final_df['Additional_Info'].unique()


# In[184]:


final_df.info()


# In[185]:


final_df['duration_hour']=final_df['Duration'].str.split(' ').str[0].str.split('h').str[0]


# In[186]:


final_df[final_df['duration_hour']=='5m']


# In[187]:


final_df.drop(6474,axis=0,inplace=True)
final_df.drop(2660,axis=0,inplace=True)


# In[191]:


# Remove non-numeric characters and convert to integers
final_df['duration_hour'] = final_df['duration_hour'].str.extract('(\d+)').astype(float).astype('Int64')

# Display the DataFrame
final_df.head()


# In[192]:


final_df.drop('Duration',axis=1,inplace=True)


# In[193]:


final_df.head(1)


# In[194]:


final_df['Airline'].unique()


# In[197]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()


# In[198]:


final_df['Airline']=labelencoder.fit_transform(final_df['Airline'])
final_df['Source']=labelencoder.fit_transform(final_df['Source'])
final_df['Destination']=labelencoder.fit_transform(final_df['Destination'])
final_df['Additional_Info']=labelencoder.fit_transform(final_df['Additional_Info'])


# In[199]:


final_df.shape


# In[200]:


final_df.head(2)


# In[201]:


final_df[['Airline']]


# In[204]:


from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Create an instance of OneHotEncoder
ohe = OneHotEncoder()

# Fit and transform the 'Airline' column into one-hot encoded vectors
airline_encoded = ohe.fit_transform(np.array(final_df['Airline']).reshape(-1,1))

# Print the shape of the resulting encoded array
print(airline_encoded.shape)


# In[205]:


final_df.head()


# In[206]:


final_df.info()


# In[209]:


import pandas as pd

# Assuming final_df contains the DataFrame and you want to encode columns "Airline", "Source", and "Destination"
encoded_df = pd.get_dummies(final_df, columns=["Airline", "Source", "Destination"], drop_first=True)

# Display the first few rows of the encoded DataFrame
print(encoded_df.head())


# In[ ]:




