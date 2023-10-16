#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('SalesForCourse_quizz_table.csv')


# In[3]:


df


# In[4]:


df.isna().mean()*100


# # Drop column1 92.6 nulls values

# In[5]:


df.drop('Column1',axis=1,inplace=True)


# In[6]:


df.drop('index',axis=1,inplace=True)


# In[7]:


df.isna().sum()


# In[8]:


df = df.dropna(axis=0)


# In[9]:


df.isna().sum()


# # Create profit column

# In[10]:


df['Profit'] = df['Revenue'] - df['Cost']


# # Create column to calculate the difference between unit cost and unit price

# In[11]:


df['diff'] = df['Unit Price'] - df['Unit Cost'] 


# # Extracting Date column

# In[12]:


df['Date'] = pd.to_datetime(df['Date'] , errors='coerce')


# In[13]:


df['DayName'] = df['Date'].dt.day_name()
df['Month'] = df['Date'].dt.month_name()
df['MonthN'] = df['Date'].dt.month
df['Year'] = df['Year'].astype(int)


# In[14]:


df.head()


# In[15]:


df.sort_values(by = 'Date' , inplace = True )


# In[16]:


df.reset_index(drop=True, inplace=True)


# In[17]:


df.info()


# # Creating seasons column 

# In[18]:


def season(MonthN):
    if MonthN in [12,1,2]:
        return 'Winter'
    elif MonthN in [3,4,5]:
        return 'Spring'
    elif MonthN in [6,7,8]:
        return 'Summer'
    else:
        return 'Autumn'


# In[19]:


df['season'] = df['MonthN'].apply(season)


# In[20]:


df.head()


# In[21]:


df.describe()


# In[22]:


df.describe(include=['O'])


# In[23]:


df[df["Profit"]<0]


# In[24]:


df["Sub Category"].value_counts()


# In[25]:


df["Country"].value_counts()


# In[26]:


df["State"].value_counts().head(10)


# In[27]:


df["State"].value_counts().sort_values(ascending=True).head(10)


# In[28]:


df["Profit"].describe()


# In[29]:


numeric_columns = df.select_dtypes(include = np.number).columns

numeric_columns


# In[30]:


numeric_columns = df.select_dtypes(include=['int64', 'float64'])  # Select only numeric columns
correlation_matrix = numeric_columns.corr()
correlation_matrix 


# In[31]:


for col in numeric_columns:
    sns.boxplot(data= df , x = col)
    plt.show()


# # swap values function

# In[32]:


def swap_values_if_needed(row):
    if row['Unit Price'] <= row['Unit Cost']:
        row['Unit Cost'], row['Unit Price'] = row['Unit Price'], row['Unit Cost']
    return row

df = df.apply(swap_values_if_needed, axis=1)
df


# # Re calculation to the columns

# In[33]:


df['Cost'] = df['Unit Cost'] * df['Quantity']


# In[34]:


df['Revenue'] = df['Unit Price'] * df['Quantity']


# In[35]:


df['Profit'] = df['Revenue'] - df['Cost']


# In[36]:


df['diff'] = df['Unit Price'] - df['Unit Cost'] 


# In[37]:


df.head()


# In[38]:


df.describe()


# # plots

# In[39]:


for col in numeric_columns:
    sns.boxplot(data= df , x = col)
    plt.show()


# # detecting outlier in unit cost

# In[40]:


outlier_unitCost=df[df["Unit Cost"]>2000]
outlier_unitCost


# In[41]:


outlier_unitCost2= outlier_unitCost[outlier_unitCost["Quantity"]<2]
outlier_unitCost2


# In[42]:


df = df.drop(outlier_unitCost2.index)


# In[43]:


df.describe()


# In[44]:


from datasist.structdata import detect_outliers
outlier_index=detect_outliers(df,0,numeric_columns)
outlier_index


# In[45]:


len(outlier_index)


# In[46]:


numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Select only numeric columns
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[47]:


category_profit = df.groupby(['Sub Category','Year']).agg({'Profit': ['mean', 'min', 'max']}).reset_index()
category_profit


# # Visuals

# In[48]:


px.pie(data_frame=df , names=df['Country'] ,title='Countries ')


# # Distribution of profit in each Country
# 

# In[49]:


px.pie(data_frame=df , names=df['Country'],values=df["Profit"])


# In[50]:


# customer ages 


# In[51]:


ages_state_profit = df.groupby(['Country'])[['Customer Age']].agg( ['mean', 'min', 'max']).reset_index()
ages_state_profit


# In[52]:


px.box(data_frame=df , x = df['Country'],y=['Customer Age'] ,title='Ages Distribution by Country')


# In[53]:


px.pie(data_frame=df , names=df['Customer Gender'] ,title='Gender')


# In[54]:


gender_state_profit = df.groupby(['Country'])[['Customer Gender']].value_counts().reset_index()
gender_state_profit


# In[55]:


px.bar(gender_state_profit, x='Country', y='count', color='Customer Gender',
           title='Gender Distribution by Country')


# # The Distribution of quantity of seasons

# In[56]:


px.pie(data_frame=df , names=df['season'],title=' Seasons')


# # The Distribution of season's profit

# In[57]:


px.pie(data_frame=df , names=df['season'],values=df["Profit"] ,title=' Seasons Profit ')


# # The Avg profit of each Sub Category in each season

# In[58]:


px.histogram(data_frame=df , x = df['season'] , y = df['Profit'] , barmode='group'  ,
             histfunc='sum' , color=df['Sub Category'] , text_auto=True )


# # The Avg Profit in each Season in each year

# In[59]:


px.histogram(data_frame=df , x = df['season'] , y = df['Profit'] , barmode='group'  ,
             histfunc='avg' , color=df['Year'] , text_auto=True )


# # Each season type of sub category product

# In[62]:


season_product = df.groupby(['Sub Category'])["season"].agg(['max', 'min']).reset_index()
season_product


# In[63]:


px.box(data_frame=df , y = df['Sub Category'],x=df['season'])


# In[64]:


ages_profit = df.groupby('Sub Category').agg({'Customer Age': ['mean', 'min', 'max']}).reset_index()
ages_profit


# In[65]:


best_selling = df.groupby(['State'])[['Sub Category']].max().reset_index()
best_selling


# In[66]:


best_selling.value_counts('Sub Category')


# In[67]:


px.box(data_frame=df , x = df['Sub Category'],y=['State'])


# # Profit of each each sub category in each season

# In[68]:


px.histogram(data_frame=df , x = df['season'] , y = df['Profit']  ,color='Sub Category' ,
             histfunc='sum' , text_auto=True )


# In[69]:


px.box(data_frame=df , x = df['Product Category'],y=['State'])


# In[70]:


px.histogram(data_frame=df , y = df['Product Category'].sort_values() , title='Quantity of products ', text_auto=True )


# In[71]:


px.pie(data_frame=df , names=df['Product Category'],values=df["Profit"])


# In[72]:


ages_state_profit = df.groupby(['Customer Gender'])[['Customer Age']].agg( ['mean', 'min', 'max']).reset_index()
ages_state_profit


# In[73]:


px.box(data_frame=df , x = df['Customer Gender'],y=['Customer Age'])


# # Distribution of Sum Profit according to Ages in each {country}

# In[76]:


State = df['State'].unique()


# In[77]:


for State in State :
    # Filter the data for the current country
    State_data = df[df['State'] == State]
    
    # Create a histogram for the current country
    fig = px.histogram(data_frame=State_data, x='Customer Age', y='Profit', color='State',
                        title=f'Distribution of Sum Profit according to Ages in {State}',
                       marginal='box', nbins=20)
   
    fig.show()


# In[78]:


df["Customer Age"].describe()


# In[79]:


countries = df['Country'].unique()


# In[80]:


for country in countries:
    # Filter the data for the current country
    country_data = df[df['Country'] == country]
    
    # Create a histogram for the current country
    fig = px.histogram(data_frame=country_data, x='Customer Age', y='Profit', color='Country',
                        title=f'Distribution of Sum Profit according to Ages in {country}',
                       marginal='box', nbins=20)
   
    fig.show()


# # The Avg profit in 2015 & 2016 For each Sub Category
# 

# In[81]:


px.histogram(data_frame=df , x = df['Sub Category'] , y = df['Profit'] , barmode='group'  ,
             histfunc='avg' , color=df['Year'] , text_auto=True )


# In[82]:


px.histogram(data_frame=df , x = df['Sub Category'] ,y=df["Profit"] , color = df['Country']
             , text_auto = True , title='Distribution of Sum Profit according to Ages in each Country', marginal='box'
             , nbins=20)


# # Sum Of Profit Monthly Each Year

# In[83]:


px.histogram(data_frame=df , y = df['Month'], x=df['Profit'] ,color=df['Year'] , title='Sum Of Profit Monthly Each Year ', text_auto=True )


# In[84]:


px.histogram(data_frame=df , y = df["Profit"], x=df['Sub Category'] ,color=df['Year'] , title='Sum Of Profit Monthly Each Year ', text_auto=True )


# # The intervel age in each state

# In[85]:


ages_state_profit = df.groupby(['Sub Category'])[['Customer Age']].agg(['min','mean', 'max']).reset_index()
ages_state_profit


# # The intervel age of each Sub Product

# In[86]:


px.box(data_frame=df , x = df['Sub Category'],y=['Customer Age'],title = "Intervel of ages in each sub product")


# In[87]:


year_2015=df[df["Year"]==2015]


# In[88]:


top2 = year_2015.groupby('MonthN')['Profit'].mean().reset_index()


# In[89]:


px.line(data_frame=top2 , x = top2.index , y = top2['Profit'] )


# In[90]:


year_2016=df[df["Year"]==2016]


# In[91]:


top3 = year_2016.groupby('MonthN')['Profit'].mean().reset_index()


# In[92]:


px.line(data_frame=top3 , x = top3.index , y = top3['Profit'] )


# # RFM Analysis

# In[93]:


import datetime as dt


# In[94]:


today_date=dt.datetime(2016,8,2)


# In[95]:


rfm=df.groupby("State").agg({"Date": lambda date:(today_date - date.max()).days,
                            "Quantity": lambda num : num.nunique(),
                            "Profit": lambda revenue : revenue.sum()})
rfm


# In[96]:


rfm.columns=["Recency","Frequency","Monetry"] 


# In[97]:


rfm = rfm[rfm["Monetry"]>0]
rfm


# In[98]:


rfm.describe().T


# In[99]:


rfm["r_score"]=pd.qcut(rfm["Recency"],4,labels=[4,3,2,1])


# In[100]:


rfm["f_score"]=pd.qcut(rfm["Frequency"].rank(method="first"),4,labels=[1,2,3,4])


# In[101]:


rfm["m_score"]=pd.qcut(rfm["Monetry"],4,labels=[1,2,3,4])


# In[102]:


rfm.head()


# In[103]:


rfm["rfm_score"]=rfm['rfm_score'] = 100 * rfm['r_score'].astype(int) + 10 * rfm['f_score'].astype(int) + rfm['m_score'].astype(int)
rfm


# In[104]:


rfm.head()


# In[105]:


def customer_segmentation(rfm_score):
    
    if rfm_score ==444:
        return 'VIP customer'
    elif rfm_score >= 443 and rfm_score < 444:
        return 'Loyal customer'
    elif rfm_score >=421 and rfm_score < 443:
        return 'become a loyal customer'
    elif rfm_score >=344 and rfm_score < 421:
        return 'recent customer'
    elif rfm_score >=323 and rfm_score < 344:
        return 'potential customer'
    elif rfm_score >=224 and rfm_score<311:
        return 'high risk to churn'
    else:
        return 'loosing customer'

rfm['customer_segmentation'] = rfm['rfm_score'].apply(customer_segmentation)
rfm


# In[106]:


plt.figure(figsize=[10,6])
sns.histplot(rfm['customer_segmentation'])
plt.xticks(rotation=90)


# In[107]:


rfm[rfm['customer_segmentation']=="VIP customer"]


# In[108]:


rfm[rfm['customer_segmentation']=="loosing customer"]


# In[109]:


a=df[df["State"]=='Arizona']
a


# In[110]:


a.describe(include=['O'])


# In[111]:


# I have to drop date here


# In[112]:


df.drop('Date',axis=1,inplace=True)


# # split data into input and output

# In[113]:


X = df.drop('Profit' , axis = 1)
y = df['Profit']


# In[114]:


X


# # dealing with categorical data

# In[115]:


X = pd.get_dummies(X , drop_first = True)
X


# # split data into train and test

# In[116]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X, y , test_size = 0.2, random_state = 42)


# In[117]:


x_train


# In[118]:


x_test


# # Scaling by RobustScaler

# In[119]:


from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
x_train_rscaler = rs.fit_transform(x_train)
x_test_rscaler = rs.transform(x_test)


# In[120]:


x_train_rscaler = pd.DataFrame(x_train_rscaler, columns = rs.get_feature_names_out())
x_train_rscaler


# In[ ]:





# In[ ]:




