import pandas as pd
from datetime import date
from datetime import timedelta
from datetime import *
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import json
import base64
import urllib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from pmdarima import auto_arima
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

raw_df = pd.read_csv("DailyDelhiClimateTrain.csv")
raw_df
if ((raw_df.duplicated()).sum() > 0):
    print("There are:", (raw_df.duplicated()).sum(), "duplicates.")
    raw_df.drop_duplicates(inplace=True)

raw_df

raw_df.info()


def convert_to_datetime(df):
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    except:
        pass

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
    except:
        pass

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y/%d/%m')
    except:
        pass

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    except:
        pass

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%Y/%m')
    except:
        pass

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y/%d')
    except:
        pass


convert_to_datetime(raw_df)
raw_df.info()
raw_df.iloc[:, 0]
raw_df.sort_values(by=['Date'], inplace=True)
df = raw_df.set_index('Date')

# Getting only the numerical data

df = df._get_numeric_data()

# Remove Duplicates

if ((df.duplicated()).sum() > 0):
    print("There are:", (df.duplicated()).sum(), "duplicates.")
    df.drop_duplicates(inplace=True)
df

print(df)

# Filling NaN values

df = df.fillna(method='ffill').fillna(method='bfill')


# Extending Dataset


def extend_dataset(forecast_days):
    index_count = 0
    new_df = df.copy()
    for column in df.columns:
        stepwise_fit = auto_arima(df[column], start_p=1, start_q=1,
                                  max_p=1, max_q=1, m=12,
                                  start_P=0, seasonal=True,
                                  d=None, D=1, trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)

        best_params = stepwise_fit.get_params()

        model = sm.tsa.statespace.SARIMAX(df[column],
                                          order=best_params["order"],
                                          seasonal_order=best_params["seasonal_order"])
        result = model.fit()

        forecast = result.predict(start=len(df),
                                  end=(len(df) - 1) + forecast_days,
                                  typ='levels').rename('Forecast')

        if (index_count == 0):
            idx = pd.date_range(df.index.max(), forecast.index.max()).union(df.index)
            new_df = df.reindex(idx)

        new_df[column] = new_df[column].fillna(forecast)
        index_count += 1
    print(new_df)


# Useful variables

num_columns = len(df.columns)
num_rows = df[df.columns[0]].count()
target_col_name = df.columns[num_columns - 1]

#

df.head()
decompose_data = seasonal_decompose(df[target_col_name], model="additive")
decompose_data.plot()


extend_dataset(50)

# Yigit Can Part

print(df)

# Implementing Models

# Support Vector Regression SVR

x = df.iloc[:, 1:2].values  # x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)

sc = StandardScaler()
x = sc.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

r = SVR(kernel="rbf")
r.fit(x, y)
print(sc_y.inverse_transform(r.predict(sc.transform([[6.5]]))))

last_col_name = df.iloc[:, -1].name
plt.scatter(sc.inverse_transform(x), sc_y.inverse_transform(y), color="red")
plt.plot(sc.inverse_transform(x), sc_y.inverse_transform(r.predict(x)), color="blue")
plt.title("SVR")
plt.xlabel("Date")
plt.ylabel(last_col_name)
plt.savefig("SVR_output.png", dpi=100)
plt.show()

# Vector Auto Regression VAR


# ARIMA


# Random Forest

regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# fit the regressor with x and y data
regressor.fit(x, y)
Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values

X_grid = np.arange(min(x), max(x), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data
plt.scatter(x, y, color='blue')

# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid),
         color='green')
plt.title('Random Forest Regression')
plt.xlabel('Date')
plt.ylabel(last_col_name)
plt.savefig("RFR_output.png", dpi=100)
plt.show()

SVR_data = {}
RFR_data = {}
with open('SVR_output.png', mode='rb') as file:
    img = file.read()
SVR_data['img'] = base64.encodebytes(img).decode('utf-8')
print(json.dumps(SVR_data))

with open('RFR_output.png', mode='rb') as file:
    img = file.read()
RFR_data['img'] = base64.encodebytes(img).decode('utf-8')

print(json.dumps(RFR_data))

print(SVR_data == RFR_data)
'''
import pymysql
import json
from flask import Flask, render_template, request, redirect, Response
app = Flask(__name__)


@app.route('/test', methods=["POST", "GET"])
def getMySQlData():
    tableData = []
    connection = pymysql.connect(host='db-auto-performancetesting',
                                 user='DBUser',
                                 password='*******',
                                 database='DbPerformanceTesting',
                                 port=3302,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:
            sql = "SELECT TestcaseName, AverageExecutionTimeInSeconds FROM PASPerformanceData WHERE BuildVersion='38.1a141'"
            cursor.execute(sql)
            while True:
                row = cursor.fetchone()
                if row == None:
                    break
                tableData.append(row)
            tableDataInJson = json.dumps(RFR_data)
            print(tableDataInJson)
            return tableDataInJson
    finally:
        connection.close()

if __name__ == "__main__":
    app.run() 
'''
