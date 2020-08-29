import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as ply
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


nyc = pd.read_csv('AB_NYC_2019.csv')

#Clean
nyc['name'] = nyc['name'].replace(np.nan,0)
nyc['host_name'] = nyc['host_name'].replace(np.nan,0)
nyc['last_review'] = nyc['last_review'].replace(np.nan,0)
nyc['reviews_per_month'] = nyc['reviews_per_month'].replace(np.nan,0)



def neighbourhood():
    groups = nyc['neighbourhood_group'].value_counts()
    groups.plot(kind='pie', autopct='%1.1f%%')
    plt.show()
    sns.countplot(x='neighbourhood_group', data=nyc)
    plt.show()
    popular = nyc['neighbourhood'].value_counts().sort_values(ascending=False)[:10].sort_values()  #Most popular
    popular.plot(kind='barh')
    plt.show()
#print(neighbourhood())

def prices():
    cheap = nyc.groupby('neighbourhood').agg({'price':'mean'}).sort_values(by='price').reset_index()  #Sort by mean
    plt.figure(figsize=(12,6))
    sns.barplot(x='price',y='neighbourhood',data=nyc.nlargest(11, ['price']))  #Expensive
    plt.show()
    sns.barplot(x='price',y='neighbourhood',data=cheap.head(10))  #Cheapest
    plt.show()
#print(prices())

def rooms():
    rooms = nyc['room_type'].value_counts()
    rooms.plot(kind='pie', autopct='%1.1f%%')
    plt.show()
    sns.countplot(x='room_type',data=nyc)
    plt.show()
    sns.countplot(x='room_type',hue='neighbourhood_group', data=nyc)
    plt.show()
#print(rooms())

def maping():
    plt.scatter(x=nyc.longitude, y=nyc.latitude,c=nyc.availability_365)
    color = plt.colorbar()
    color.set_label('Availability')
    plt.show()
    plt.scatter(x=nyc['longitude'], y=nyc['latitude'], c=nyc['price'])
    bar = plt.colorbar()
    bar.set_label('price($)')
    plt.show()
    sns.scatterplot(x=nyc['longitude'], y=nyc['latitude'], hue=nyc['neighbourhood_group'])
    plt.show()
    sns.scatterplot(x=nyc['longitude'],y=nyc['latitude'], hue=nyc['room_type'])
    plt.show()
#print(maping())

def priceDistribution():
    lower = nyc[nyc['price'] < 500]
    sns.boxplot(x='neighbourhood_group',y='price',data=lower)
    plt.show()
    upper = nyc[nyc['price'] > 500]
    sns.boxplot(x='neighbourhood_group',y='price',data=upper)
    plt.show()
    sns.scatterplot(y=nyc['price'], x=nyc['minimum_nights'])
    plt.show()
    sns.scatterplot(y=nyc['price'], x=nyc['availability_365'])
    plt.show()
#print(priceDistribution())

def nights():
    nights = nyc[(nyc['minimum_nights'] <= 30) & (nyc['minimum_nights'] > 0)]['minimum_nights']
    sns.distplot(nights,bins=30)
    plt.show()
#print(nights())

def meanPrice():
    price = nyc.groupby('neighbourhood_group').mean()['price'].reset_index().sort_values('price',ascending=False)
    sns.barplot(x='neighbourhood_group',y='price',data=price)
    plt.show()
#print(meanPrice())

#sns.heatmap(nyc.corr(), square=True,annot=True)
#plt.show()
nyc.drop(['id','name','host_name','last_review'],axis=1,inplace=True)

encode = LabelEncoder()
encode.fit(nyc['neighbourhood'])
nyc['neighbourhood'] = encode.transform(nyc['neighbourhood'])

encode = LabelEncoder()
encode.fit(nyc['neighbourhood_group'])
nyc['neighbourhood_group'] = encode.transform(nyc['neighbourhood_group'])

encode = LabelEncoder()
encode.fit(nyc['room_type'])
nyc['room_type'] = encode.transform(nyc['room_type'])

nyc.sort_values(by='price',ascending=True,inplace=True)
#print(nyc.head())

x = nyc.drop(['price'], axis=1)
y = nyc['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
lin = LinearRegression(n_jobs=None,fit_intercept=True,normalize=False,copy_X=True)
lin.fit(x_train,y_train)
predict = lin.predict(x_test)
#print("mse:", np.sqrt(mean_squared_error(y_test,predict)))


def Predict():
    error = pd.DataFrame({'Actual': np.array(y_test).flatten(),
                          'Predicted': predict.flatten()}).head(20)
    # print(error)
    figure = go.Figure(data=[go.Bar(name='Predicted', x=error.index, y=error['Predicted']),
                             go.Bar(name='Actual', x=error.index, y=error['Actual'])])
    figure.show()
#print(Predict())


dec = DecisionTreeRegressor(min_samples_leaf=1, min_samples_split=2)
dec.fit(x_train,y_train)
dec_predict = dec.predict(x_test)

def DecPredict():
    dec_error = pd.DataFrame({'Actual': np.array(y_test).flatten(),
                              'Predicted': dec_predict.flatten()}).head(10)
    figure = go.Figure(data=[go.Bar(name='Predicted', x=dec_error.index, y=dec_error['Predicted']),
                             go.Bar(name='Actual', x=dec_error.index, y=dec_error['Actual'])])
    figure.show()

