# finalize model and make a prediction for monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import xgboost as xgb
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt

# transform a time series dataset into a supervised learning dataset
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


def calculateRMSE(actual_values, predicted_values):
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    return rmse

def timeinterval(csvfile, nadditions):
    firstdate = csvfile.index[0]
    seconddate = csvfile.index[1]
    diff = seconddate - firstdate
    lastdate = csvfile.index[-1]
    listofdates = []
    for index in range(nadditions):
        added = lastdate + diff
        listofdates.append(added)
        lastdate = listofdates[index]
    return listofdates

def convert_to_datetime(df):
    try:
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y/%d/%m')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%d/%Y/%m')
    except:
        pass

    try:
        df['date'] = pd.to_datetime(df['date'], format='%m/%Y/%d')
    except:
        pass

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, n_out):
        cols.append(df.shift(-i))

    agg = concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg.values


def forecast_algorithm(model, future_day_amount, csvfilename):
    # load the dataset
    data = read_csv('C://Users//Admin//Desktop//mm//' + csvfilename, header=0)
    data.rename(columns = {list(data)[0]:'date'}, inplace=True)
    convert_to_datetime(data)
    data = data.set_index('date')
    next_days = timeinterval(data, future_day_amount)
    values = data.values
    train = series_to_supervised(values, n_in=8)


    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    row = values[-8:].flatten()
    forecasted_values = []
    result = []
    if (model == "rfr"):
        model = RandomForestRegressor(n_estimators=1000)
        model.fit(trainX, trainy)

        for i in range(future_day_amount):
            yhat = model.predict(asarray([row]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))

        error_model = RandomForestRegressor(n_estimators=1000)
        error_model.fit(trainX[0:int(len(trainX) * 0.8)], trainy[0:int(len(trainy) * 0.8)])
        preds = []

        for i in range(int(len(trainX) * 0.2)):
            pred_y = error_model.predict(asarray([values[int(len(trainX) * 0.8)+i : int(len(trainX) * 0.8)+i+8].flatten()]))
            preds.append(pred_y)

        for i in range(future_day_amount):
            result.append((forecasted_values[i], str(next_days[i])[0:10]))

        rmse_error = calculateRMSE(trainy[int(len(trainy) * 0.8)+1:], preds)

    if (model == "svr"):

        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=250)
        regr = make_pipeline(StandardScaler(), svr_rbf)
        regr.fit(trainX, trainy)

        for i in range(future_day_amount):
            yhat = regr.predict(asarray([row]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))

        kernell = "rbf"
        if all(element == forecasted_values[0] for element in forecasted_values):
            kernell = "linear"
            forecasted_values.clear()
            row = values[-8:].flatten()
            svr_line = SVR(kernel='linear', C=1e3, gamma=250)
            regr = make_pipeline(StandardScaler(), svr_line)
            regr.fit(trainX, trainy)
            for i in range(future_day_amount):
                yhat = regr.predict(asarray([row]))
                # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
                row = np.delete(row, 0)
                row = np.append(row, yhat)
                forecasted_values.append(float(yhat))

        error_model = SVR(kernel=kernell, C=1e3, gamma=250)
        regr_error = make_pipeline(StandardScaler(), error_model)
        regr_error.fit(trainX[0:int(len(trainX) * 0.8)], trainy[0:int(len(trainy) * 0.8)])
        preds = []

        for i in range(int(len(trainX) * 0.2)):
            pred_y = regr_error.predict(
                asarray([values[int(len(trainX) * 0.8) + i: int(len(trainX) * 0.8) + i + 8].flatten()]))
            preds.append(pred_y)

        for i in range(future_day_amount):
            result.append((forecasted_values[i], str(next_days[i])[0:10]))

        rmse_error = calculateRMSE(trainy[int(len(trainy) * 0.8) + 1:], preds)


    if (model == "dtr"):
        dt_reg = DecisionTreeRegressor(max_depth=22, max_features="auto", random_state=10)

        dt_reg.fit(trainX, trainy)

        for i in range(future_day_amount):
            yhat = dt_reg.predict(asarray([row]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))

        error_model = dt_reg = DecisionTreeRegressor(max_depth=22, max_features="auto", random_state=10)
        error_model.fit(trainX[0:int(len(trainX) * 0.8)], trainy[0:int(len(trainy) * 0.8)])
        preds = []

        for i in range(int(len(trainX) * 0.2)):
            pred_y = error_model.predict(asarray([values[int(len(trainX) * 0.8) + i: int(len(trainX) * 0.8) + i + 8].flatten()]))
            preds.append(pred_y)

        for i in range(future_day_amount):
            result.append((forecasted_values[i], str(next_days[i])[0:10]))

        rmse_error = calculateRMSE(trainy[int(len(trainy) * 0.8) + 1:], preds)

    if (model == "xgb"):
        regressor = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
        )
        regressor.fit(trainX, trainy)

        for i in range(future_day_amount):
            yhat = regressor.predict(asarray([row]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))

        error_model = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
        )
        error_model.fit(trainX[0:int(len(trainX) * 0.8)], trainy[0:int(len(trainy) * 0.8)])
        preds = []

        for i in range(int(len(trainX) * 0.2)):
            pred_y = error_model.predict(asarray([values[int(len(trainX) * 0.8) + i: int(len(trainX) * 0.8) + i + 8].flatten()]))
            preds.append(pred_y)

        for i in range(future_day_amount):
            result.append((forecasted_values[i], str(next_days[i])[0:10]))

        rmse_error = calculateRMSE(trainy[int(len(trainy) * 0.8) + 1:], preds)

    '''
    if (model == "srm"):
        index_count = 0
        new_df = series.copy()
        for column in series.columns:
            stepwise_fit = auto_arima(series[column], start_p=1, start_q=1,
                                      max_p=1, max_q=1, m=12,
                                      start_P=0, seasonal=True,
                                      d=None, D=1, trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)

            best_params = stepwise_fit.get_params()

            model = SARIMAX(series[column],
                                              order=best_params["order"],
                                              seasonal_order=best_params["seasonal_order"])
            result = model.fit()

            forecast = result.predict(start=len(series),
                                      end=(len(series) - 1) + future_day_amount,
                                      typ='levels').rename('Forecast')
            forecasted_values.append(float(forecast))
    # s
    '''

    past_and_forecasted_values = []
    past_values = []
    for i in range(len(values)):
        past_and_forecasted_values.append(float(values[i]))
        past_values.append(float(values[i]))

    for i in range(len((forecasted_values))):
        past_and_forecasted_values.append(forecasted_values[i])

    col = []

    for i in range(0, len(past_and_forecasted_values)):
        if i < len(past_values):
            col.append('blue')
        else:
            col.append('r')

    for i in range(len(past_and_forecasted_values)):
        # plotting the corresponding x with y
        # and respective color
        plt.scatter(i, past_and_forecasted_values[i], c=col[i], s=10,
                    linewidth=0)

    plt.ylabel("Prices")
    plt.xlabel("Data points")
    plt.title("Electricity Price Forecasting")

    plt.show()

    plt.plot(past_values)
    plt.ylabel("Prices")
    plt.xlabel("Data points")
    plt.title("Electricity Price Forecasting - Row data")
    plt.show()

    plt.plot(past_and_forecasted_values)
    plt.ylabel("Prices")
    plt.xlabel("Data points")
    plt.title("Electricity Price Forecasting - Forecasted data")
    plt.show()
    print("Rmse:", rmse_error)
    return result, rmse_error


def graphVal(name):
    a = name
    print(a)


forecasted_values = forecast_algorithm("svr", 50, "Electric_cost.csv")
print(forecasted_values)
'''
past_and_forecasted_values = []
past_values = []
for i in range(len(values)):
    past_and_forecasted_values.append(float(values[i]))
    past_values.append(float(values[i]))

for i in range(len((forecasted_values))):
    past_and_forecasted_values.append(forecasted_values[i])

col = []

for i in range(0, len(past_and_forecasted_values)):
    if i < len(past_values):
        col.append('blue')
    else:
        col.append('r')

for i in range(len(past_and_forecasted_values)):
    # plotting the corresponding x with y
    # and respective color
    plt.scatter(i, past_and_forecasted_values[i], c=col[i], s=10,
                linewidth=0)





plt.ylabel("Prices")
plt.xlabel("Data points")
plt.title("Electricity Price Forecasting")

plt.show()


plt.plot(past_values)
plt.ylabel("Prices")
plt.xlabel("Data points")
plt.title("Electricity Price Forecasting - Row data")
plt.show()

plt.plot(past_and_forecasted_values)
plt.ylabel("Prices")
plt.xlabel("Data points")
plt.title("Electricity Price Forecasting - Forecasted data")
plt.show()
'''
