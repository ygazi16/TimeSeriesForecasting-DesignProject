from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor


def transform_to_supervised(data, previous_steps=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(previous_steps, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, n_out):
        cols.append(df.shift(-i))

    agg = pd.concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg.values

data = pd.read_csv('DailyDelhiClimate.csv',header=0, index_col=0)
values = data.values


x = transform_to_supervised(data, previous_steps=4)


trainX, trainy = x[:, :-1], x[:, -1]

new_trainX = []
for elt in trainX:
    new_trainX.append([elt[index] for index in [1,2,3,5,6,7,9,10,11,13,14,15] ])

new_trainX = np.array(new_trainX)


def forecast_algorithm(model, future_day_amount):
    row = values[-4:, :-1].flatten()
    forecasted_values = []

    if(model == "RandomForest"):
        model = RandomForestRegressor(n_estimators=1000)
        model.fit(new_trainX, trainy)

        for i in range(future_day_amount):
            yhat = model.predict(asarray([row]))
            # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))


    if(model =="xgb"):
        regressor = xgb.XGBRegressor(
                    n_estimators=100,
                    reg_lambda=1,
                    gamma=0,
                    max_depth=3
                )
        regressor.fit(new_trainX, trainy)

        for i in range(future_day_amount):
            yhat = regressor.predict(asarray([row]))
            # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))
#
    if (model == "svr"):
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=250)
        svr_rbf.fit(new_trainX, trainy)

        for i in range(future_day_amount):
            yhat = svr_rbf.predict(asarray([row]))
            # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))

    if (model == "lgb"):
        d_train = lgb.Dataset(new_trainX, label=trainy)
        params = {'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'binary_logloss',
                  'sub_feature': 0.5,
                  'num_leaves': 10,
                  'min_data': 50,
                  'max_depth': 10}
        lgb_model = lgb.train(params, d_train, num_boost_round=100)


        for i in range(future_day_amount):
            yhat = lgb_model.predict(asarray([row]))
            # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))

    if (model == "decision tree"):
        ka_reg = DecisionTreeRegressor(max_depth=1)

        ka_reg.fit(new_trainX, trainy)

        for i in range(future_day_amount):
            yhat = ka_reg.predict(asarray([row]))
            # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))

    return forecasted_values

forecasted_values = forecast_algorithm("xgb", 50)
print("forecasted değerler")
print(forecasted_values)


past_and_forecasted_values = []
past_values = []
for i in range(len(values)):
    past_and_forecasted_values.append(float(values[i][len(values[i])-1]))
    past_values.append(float(values[i][len(values[i])-1]))

for i in range(len(forecasted_values)):
    past_and_forecasted_values.append(forecasted_values[i])


col = []

for i in range(0, len(past_and_forecasted_values)):
    if i < len(past_values):
        col.append('blue')
    else:
        col.append('r')

for i in range(len(past_and_forecasted_values)):

    plt.scatter(i, past_and_forecasted_values[i], c=col[i], s=10,
                linewidth=0)


plt.ylabel("Mean Temperatures")
plt.xlabel("Data points")
plt.title("Mean Temperature Forecasting")
plt.show()



plt.plot(past_values)
plt.ylabel("Prices")
plt.xlabel("Data points")
plt.title("Mean Temperature Forecasting - Row Data")
plt.show()

plt.plot(past_and_forecasted_values)
plt.ylabel("Mean Temperatures")
plt.xlabel("Data points")
plt.title("Mean Temperature Forecasting - Forecasted Data")
plt.show()