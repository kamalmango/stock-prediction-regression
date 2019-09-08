import pandas as pd
import datetime as dt
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics

ticker = "DIS"
start_date = dt.date.today() - dt.timedelta(365)
end_date = dt.date.today()
data = pdr.get_data_yahoo(ticker, start_date, end_date)

data.head()

data['Adj Close'].plot(label='DIS', figsize=(16,8), title='Adjusted Closing Price', grid=True)

timestamps = data.index.tolist()
prices = data['Adj Close'].tolist()
dates = []


for date in timestamps:
  dates.append(int(date.strftime("%d-%m-%Y").split('-')[0]))

prices = np.array(prices).reshape(-1,1)
dates = np.array(dates).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(dates, prices, test_size=0.2, shuffle=False)

# Linear Regression
regressor = LinearRegression()  
regressor.fit(x_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

y_pred_train_linear = regressor.predict(x_train)
y_pred_test_linear = regressor.predict(x_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred_test_linear.flatten()})

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

df_linear = data.copy()
df_linear.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_linear = df_linear.iloc[:200, :]
df_linear['Adj Close Train'] = y_pred_train_linear
df_linear.plot(label='DIS', figsize=(16,8), title='Adjusted Closing Price', grid=True)

df_linear = data.copy()
df_linear.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_linear = df_linear.iloc[-50:]
df_linear['Adj Close Test'] = y_pred_test_linear
df_linear.plot(label='DIS', figsize=(16,8), title='Adjusted Closing Price', grid=True)

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(x_train, y_train)

y_pred_train_ridge = ridge_model.predict(x_train)
y_pred_test_ridge = ridge_model.predict(x_test)

df_ridge = data.copy()
df_ridge.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_ridge = df_ridge.iloc[:200, :] 
df_ridge['Adj Close Train'] = y_pred_train_ridge
df_ridge.plot(label='DIS', figsize=(16,8), title='Adjusted Closing Price', grid=True)

df_ridge = data.copy()
df_ridge.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_ridge = df_ridge.iloc[-50:] 
df_ridge['Adj Close Test'] = y_pred_test_ridge
df_ridge.plot(label='DIS', figsize=(16,8), title='Adjusted Closing Price', grid=True)

#Lasso
lasso_model = Lasso()
lasso_model.fit(x_train, y_train)

y_pred_train_lasso = ridge_model.predict(x_train)
y_pred_test_lasso = ridge_model.predict(x_test)

df_lasso = data.copy()
df_lasso.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_lasso = df_lasso.iloc[:200, :] 
df_lasso['Adj Close Train'] = y_pred_train_lasso
df_lasso.plot(label='DIS', figsize=(16,8), title='Adjusted Closing Price', grid=True)

df_lasso = data.copy()
df_lasso.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_lasso = df_lasso.iloc[-50:] 
df_lasso['Adj Close Test'] = y_pred_test_lasso
df_lasso.plot(label='DIS', figsize=(16,8), title='Adjusted Closing Price', grid=True)









