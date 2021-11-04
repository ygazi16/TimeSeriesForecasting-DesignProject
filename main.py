import pandas as pd
from datetime import date
from datetime import timedelta
from datetime import *
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR

df = pd.read_csv("../../Desktop/energy_data.csv")

df.info()

del df["NOTES"]

df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y')
df['START TIME'] = pd.to_datetime(df['START TIME'], format='%H:%M')
df['END TIME'] = pd.to_datetime(df['END TIME'], format='%H:%M')
df['START TIME'] = df['START TIME'].dt.strftime('%H:%M:%S')
df['END TIME'] = df['END TIME'].dt.strftime('%H:%M:%S')
df.sort_values(by=['DATE', "START TIME"], inplace=True)
filled_column = pd.concat([df.ffill(), df.bfill()]).groupby(level=0).mean()
df["USAGE"] = filled_column
if (df.duplicated()).sum() > 0:
    print("There are:", (df.duplicated()).sum(), "duplicates.")
    df.drop_duplicates(inplace=True)

Q1 = np.percentile(df['USAGE'], 3, interpolation='midpoint')

Q3 = np.percentile(df['USAGE'], 97, interpolation='midpoint')
IQR = Q3 - Q1

# Upper bound
upper = np.where(df['USAGE'] >= (Q3 + 1.5 * IQR))
# Lower bound
lower = np.where(df['USAGE'] <= (Q1 - 1.5 * IQR))

print("There are:", len(upper[0]), "outliers in upper bound.")
print("There are:", len(lower[0]), "outliers in lower bound.")

print("Max value for Usage is:", df["USAGE"].max())
print("Min value for Usage is:", df["USAGE"].min())

scaler = MinMaxScaler()
df['Usage_scaled'] = scaler.fit_transform(df['USAGE'].values.reshape(-1, 1))

df.info()

print(df['DATE'].ndim)

X = df.iloc[:, 1:2].values
Y = df.iloc[:, 6].values


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y)
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Support Vector Regression')
plt.xlabel('X')
plt.ylabel('Cost')
plt.show()