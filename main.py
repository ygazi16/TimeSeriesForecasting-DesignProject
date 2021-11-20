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
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from math import sqrt
from sklearn.metrics import mean_squared_error

raw_df = pd.read_csv("wind_data.csv")
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

# combining data set
numeric_df = raw_df._get_numeric_data()
df = pd.concat([raw_df.iloc[:, 0], numeric_df], axis=1)
df

if ((df.duplicated()).sum() > 0):
    print("There are:", (df.duplicated()).sum(), "duplicates.")
    df.drop_duplicates(inplace=True)
df
df.sort_values(by=['Date'], inplace=True)
print(df)

df = df.fillna(method='ffill').fillna(method='bfill')

# # Implementing Models

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(x)
print(y)

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

import json
import base64

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
