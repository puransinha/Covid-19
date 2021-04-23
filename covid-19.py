
# # This notebook tracks and Analyse the spread of the coronavirus(COVID-19) 🦠.
# 

# > ***Please let me know if there is any scope of improvement in the output or something. I don't have much experience in data science, but I dedicated my best in this.Thank you.*** 

# # * > IMPORTING MODULLES****

get_ipython().system('pip install plotly')


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import ensemble
# Input data files are available in the read-only "../input/" directory


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

train = pd.read_csv(r'D:\Assignments\Covid\train.csv', low_memory=False)
test = pd.read_csv(r'D:\Assignments\Covid\test.csv', low_memory=False)
# sample = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
sample = pd.read_csv(r'D:\Assignments\Covid\submission.csv', low_memory=False)

train.isnull().sum()
train.County.count()
train.Province_State.count()

test.isnull().sum()
test.County.count()
test.Province_State.count()

sample

sample['TargetValue'].sum()

train.sort_values(by=['TargetValue'])

# ****************** Data Visualization *****************
# fig = px.pie(train, values='TargetValue', names='Target')
# fig.update_traces(textposition='inside')
# fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

# fig.show()


fig = px.pie(train, values='TargetValue', names='Country_Region')
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.show()


fig = px.treemap(train, path=['Country_Region'], values='TargetValue',
                  color='Population', hover_data=['Country_Region'],
                  color_continuous_scale='matter', title='Current share of Worldwide COVID19 Confirmed Cases')
fig.show()


last_date = train.Date.max()

df_countries = train[train['Date']==last_date]
df_countries = df_countries.groupby('Country_Region', as_index=False)['TargetValue'].sum()
df_countries = df_countries.nlargest(10,'TargetValue')
df_countries

df_trend = train.groupby(['Date','Country_Region'], as_index=False)['TargetValue'].sum()
df_trend = df_trend.merge(df_countries, on='Country_Region')
df_trend.rename(columns={'Country_Region':'Country', 'TargetValue_x':'Cases'}, inplace=True)
df_trend

px.line(df_trend, x='Date', y='Cases', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries')

# # Data Preprocessing

#  We would drop some features Who have many Null values and not that much important.

train = train.drop(['County','Province_State','Country_Region','Target'],axis=1)
test = test.drop(['County','Province_State','Country_Region','Target'],axis=1)
train
test

from sklearn.preprocessing import OrdinalEncoder

def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df

def train_dev_split(df, days):
    #Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]

test_date_min = test['Date'].min()
test_date_max = test['Date'].max()


def avoid_data_leakage(df, date=test_date_min):
    return df[df['Date']<date]

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

train['Date']=pd.to_datetime(train['Date'])
test['Date']=pd.to_datetime(test['Date'])

test['Date']=test['Date'].dt.strftime("%Y%m%d")
train['Date']=train['Date'].dt.strftime("%Y%m%d").astype(int)


# # USING REGRESSOR TO FIND TARGET VALUES
from sklearn.model_selection import train_test_split

predictors = train.drop(['TargetValue', 'Id'], axis=1)
target = train["TargetValue"]
X_train, X_test, y_train, y_test = train_test_split(
    predictors, target, test_size = 0.22, random_state = 0)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
model = RandomForestRegressor(n_jobs=-1)
estimators = 100
model.set_params(n_estimators=estimators)

scores = []

pipeline = Pipeline([('scaler2' , StandardScaler()),
                        ('RandomForestRegressor: ', model)])
pipeline.fit(X_train , y_train)


pipeline.fit(X_train, y_train)
scores.append(pipeline.score(X_test, y_test))

X_test
test.drop(['ForecastId'],axis=1,inplace=True)
test.index.name = 'Id'
test

y_pred2 = pipeline.predict(X_test)
y_pred2

predictions = pipeline.predict(test)

pred_list = [int(x) for x in predictions]

output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})
print(output)

output


# # Finding Quantile values from the output.

a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()

a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05'].clip(0,10000)
a['q0.5']=a['q0.5'].clip(0,10000)
a['q0.95']=a['q0.95'].clip(0,10000)
a

a['Id'] =a['Id']+ 1
a

# # Submission

sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.head()

sub.to_csv("submission.csv",index=False)

# # Will try to update kernel with much better score than before.

