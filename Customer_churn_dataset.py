#!/usr/bin/env python
# coding: utf-8

# # Telco Customer Churn

# # Context
# "Predict behavior to retain customers. Our goal is to analyze all relevant customer data and develop focused customer retention programs."

# # Content
# The data set includes information about:
# 
# 1. Customers who left within the last month – the column is called Churn
# 2. Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# 3. Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges.
# 4. Demographic info about customers – gender, age range, and if they have partners and dependents

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as po
import plotly


# In[3]:


df = pd.read_csv('customer_churn.csv')


# In[4]:


df.head()


# In[6]:


print(list(df.columns))
print(len(df.columns))
print(df.shape)


# In[7]:


df.isnull().sum()


# In[8]:


df.count()


# In[9]:


df.isna().count()


# In[13]:


df.dtypes


# 
# Here we can see that we have lots of items having data types object.

# # EDA

# In[14]:


df['Churn']


# In[16]:


df['Churn'].unique()


# 1. Here we can say that churn have positive or negative values .
# 2. Yes means customer will churn out and no means customer will not churn out.
# 3. So, our first task is to convert the yes/no values to 1/0 values so that is will be easier for us to do the further operations.

# In[17]:


df['Churn'] = df['Churn'].replace({'Yes': 1 , 'No' :0})


# In[19]:


df['Churn'].head()


# let's convert replace the "No internet service" string to "No" in the following columns below.

# In[22]:


cols = ['OnlineBackup' , 'StreamingMovies', 'DeviceProtection', 'TechSupport' , 'OnlineSecurity' , 'StreamingTV']
for values in cols:
    df[values] = df[values].replace({'No internet service' :' No'})


# In[23]:


df


# In[24]:


pd.set_option('display.max_rows' ,None)
pd.set_option('display.max_columns' , None)


# In[25]:


df


# In[26]:


df['TotalCharges'] = df['TotalCharges'].replace(' ' , np.nan)


# In[29]:


df.isna().sum()


# In[30]:


# Droping the null values 

df = df[df['TotalCharges'].notnull()]
df = df.reset_index()[df.columns]


# In[31]:


df['TotalCharges'] = df['TotalCharges'].astype(float)


# In[33]:


df.shape


# # Data Visualization 

# In[37]:


df['Churn'].value_counts()


# In[38]:


## what percentage of customer have churn or not 


# In[40]:


churn_x = df['Churn'].value_counts().keys().tolist()
churn_y = df['Churn'].value_counts().values.tolist()


# In[46]:


fig = px.pie(df, labels= churn_x , values= churn_y , color_discrete_sequence = ['grey' , 'teal'],hole = 0.6) 

fig.update_layout(title='Customer Churn ', template = 'plotly_dark')

fig.show()


# In[48]:


df.groupby('gender').Churn.mean().reset_index()


# In[49]:


df.groupby('gender').MonthlyCharges.mean().reset_index()


# # Male or Female who has churn the most 

# In[61]:


gender_chunk = df.groupby('gender').Churn.mean().reset_index()

fig = go.Figure(data=[go.Bar(
            x=gender_chunk['gender'], y=gender_chunk['Churn'],
            textposition='auto',
            width=[0.2,0.2],
            marker = dict(color=['brown','purple']))])


# In[63]:


fig.update_layout(
    title='Churn rate by Gender',
    xaxis_title="Gender",
    yaxis_title="Churn rate",
        template='plotly'

)
fig.show()


# ## Does tech support plays a important role

# In[83]:


df.groupby('TechSupport').Churn.mean().reset_index()


# In[88]:


gender_chunk = df.groupby('TechSupport').Churn.mean().reset_index()

fig = go.Figure(data=[go.Bar(
            x=gender_chunk['TechSupport'], y=gender_chunk['Churn'],
            textposition='auto',
            width=[0.4,0.4],
            marker = dict(color=['brown','purple']))])


# In[89]:


fig.update_layout(
    title='Churn rate by Tech Supoort',
    xaxis_title="Tech Support",
    yaxis_title="Churn rate",
        template='plotly')

fig.show()


# ## Telecome company is not satisfying their customer with internet support?

# In[77]:


df['InternetService'].unique()


# In[81]:


internet_chunk = df.groupby('InternetService').Churn.mean().reset_index()

fig = go.Figure(data=[go.Bar(
            x=internet_chunk['InternetService'], y=internet_chunk['Churn'],
            textposition='auto',
            width=[0.4,0.2,0.2],
            marker = dict(color=['tomato','tan','cyan']))])


# In[82]:


fig.update_layout(
    title='Churn rate by Internet Services',
    xaxis_title="Internet Services",
    yaxis_title="Churn rate",
        template='plotly_dark'

)
fig.show()


# ## Payment Method, Any role in Churning??

# In[90]:


payment_chunk = df.groupby('PaymentMethod').Churn.mean().reset_index()

fig = go.Figure(data=[go.Bar(
            x=payment_chunk['PaymentMethod'], y=payment_chunk['Churn'],
            textposition='auto',
            width=[0.2,0.2,0.2,0.2],
            marker = dict(color=['teal','thistle','lime','navy']))])


# In[92]:


fig.update_layout(
    title='Churn rate by Payment Method',
    xaxis_title="Payment Method Churns",
    yaxis_title="Churn rate",
        template='plotly'

)
fig.show()


# ## Observation :
# Here we can clearly see that customers making payment through electronic check seem to be more churn than others.

# # Signing long term or short contract term churn the most?

# In[93]:


contract_chunk =df.groupby('Contract').Churn.mean().reset_index()

fig = go.Figure(data=[go.Bar(
            x=contract_chunk['Contract'], y=contract_chunk['Churn'],
            textposition='auto',
            width=[0.2,0.2,0.2,0.2],
            marker = dict(color=['teal','thistle','purple']))])


# In[94]:


fig.update_layout(
    title='Churn rate by Contract',
    xaxis_title="Contract Churns",
    yaxis_title="Churn rate",
        template='plotly_dark'

)
fig.show()


# ## Oberavtion :
# From the above bar chart we can clearly see that customers who have monthly contract seems more likely to churn as compared to one year of two year contract signed customers.

# In[95]:


ten_chunk = df.groupby('tenure').Churn.mean().reset_index()

fig = go.Figure(data=[go.Scatter(
    x=ten_chunk['tenure'],
    y=ten_chunk['Churn'],
        mode='markers',
        name='Low',
        marker= dict(size= 5,
            line= dict(width=0.8),
            color= 'yellow'
           ),
)])
fig.update_layout(
    title='Churn rate by Tenure',
    xaxis_title="Tenure",
    yaxis_title="Churn rate",
    template='plotly_dark'

)
fig.show()


# ## Observation :
# From scatter plot it is clearly visble that the smaller the contract the higer the churning rate and vice-versa 

# In[96]:


tot_chunk = df.groupby('TotalCharges').Churn.mean().reset_index()

fig = go.Figure(data=[go.Scatter(
    x=tot_chunk['TotalCharges'],
    y=tot_chunk['Churn'],
        mode='markers',
        name='Low',
        marker= dict(size= 5,
            line= dict(width=0.8),
            color= 'red'
           ),
)])


# In[98]:


fig.update_layout(
    title='Churn rate by TotalCharges',
    xaxis_title="TotalCharges",
    yaxis_title="Churn rate",
    template='plotly'

)
fig.show()


# ## Observation :
# from the figure we can see that the more the charges paid by the person the higher the chances of churning 

# In[ ]:





# In[100]:


churn_data = pd.get_dummies(df, columns = ['Contract','Dependents','DeviceProtection','gender',
                                                        'InternetService','MultipleLines','OnlineBackup',
                                                        'OnlineSecurity','PaperlessBilling','Partner',
                                                        'PaymentMethod','PhoneService','SeniorCitizen',
                                                        'StreamingMovies','StreamingTV','TechSupport'],
                              )


# In[102]:


churn_data.head()


# In[103]:


df.corr()


# # Modelling 

# In[104]:


sklearn.preprocessing import StandardScaler


# In[105]:


# Feature Scaling on 'tenure', 'MonthlyCharges', 'TotalCharges' in order to bring them on same scale
standard = StandardScaler()
columns_for_ft_scaling = ['tenure' , 'MonthlyCharges' , 'TotalCharges']


# In[106]:


churn_data[columns_for_ft_scaling] = standard.fit_transform(churn_data[columns_for_ft_scaling])


# In[107]:


churn_data.head()


# ## Observation :
# Here we can see that tenure,MonthlyCharges and TotalCharges are in same range , and the suffix are converted into subcolumns 
# 
# eg: Contract is expanded to contract_one year and Contract_Two year. These are only the individual values. Let's see how many columns we have in our dataset.

# In[108]:


list(churn_data)


# We can clearly see that we have added multiple columns for the individual values.

# In[109]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix
from sklearn.metrics import classification_report


# In[111]:


x = churn_data.drop(['Churn' ,'customerID'] , axis=1)
Y = churn_data['Churn']


# Using the train_test_split() function to split our data into training and test sets. I will use 70% of our data for training our model and 30% for testing our model.

# In[113]:


x_train,x_test,Y_train,Y_test = train_test_split(x,Y,test_size=0.3,random_state=0)


# In[114]:


x_train.shape,Y_train.shape,x_test.shape,Y_test.shape


# In[115]:


x_train


# # Logistic Regression

# In[118]:


#using logistic regression
log = LogisticRegression()

#fitting our data
log.fit(x_train,Y_train)

#making prediction
Y_pred = log.predict(x_test)


# In[120]:


Y_pred=pd.DataFrame(Y_pred)


# In[124]:


print("{:.2f}%".format(accuracy_score(Y_pred,Y_test)*100))


# # Decision tree 

# In[131]:


#using decision tree classifier
dec = DecisionTreeClassifier()

#fitting our data
dec.fit(x_train,Y_train)

#predicting the values
Y_dec_pred = dec.predict(x_test)


# In[132]:


print("{:.2f}%".format(accuracy_score(Y_dec_pred,Y_test)*100))


# # Random Forest  

# In[133]:


#using random forest classifier
rand = RandomForestClassifier()

#fitting the data
rand.fit(x_train,Y_train)

#predicting values
Y_rand_pred = rand.predict(x_test)


# In[134]:


Y_rand_pred


# In[137]:


print('{:.2f}%'.format(accuracy_score(Y_rand_pred,Y_test)*100))


# # Confusion matrix

# In[155]:


print("Logistic Regression Model")
plot_confusion_matrix(log,x_test,Y_test,cmap='Oranges')


# In[ ]:





# In[156]:


print("Decison Tree Model")
plot_confusion_matrix(dec,x_test,Y_test,cmap='Dark2');


# In[158]:


print("Random Forest Model")
plot_confusion_matrix(rand,x_test,Y_test,cmap='cividis');


# Let's predict the probability of churn for each customer using logistic regression model.

# In[ ]:





# In[160]:


churn_data["Customer Churning Probability"] = log.predict_proba(churn_data[x_test.columns])[:,1]


# In[161]:


#Created a new column in our dataset named Customer Churning Probability.


# In[162]:


churn_data[['customerID', 'Customer Churning Probability']]


# In[165]:


print(classification_report(Y_pred,Y_test))


# In[166]:


print(classification_report(Y_dec_pred,Y_test))


# In[167]:


print(classification_report(Y_rand_pred,Y_test))


# From the above probablity report it can be seen the chances of person churning or not in future 

# In[ ]:




