from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings("ignore")


def timeinterval(csvfile, nadditions):
    firstdate = csvfile.index[0]
    seconddate = csvfile.index[1]
    diff = seconddate - firstdate
    print(firstdate)
    print(seconddate)
    print("Difference as days: ", diff)
    lastdate = csvfile.index[-1]
    print("last date: ", lastdate)
    listofdates = []
    for index in range(nadditions):
        added = lastdate + diff
        listofdates.append(added)
        lastdate = listofdates[index]
    return firstdate, listofdates


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


data = read_csv("DailyDelhiClimate.csv", header=0)
convert_to_datetime(data)
values = data.values
data = data.set_index('date')
dt = timeinterval(data, 50)


x = transform_to_supervised(data, previous_steps=3)

trainX, trainy = x[:, :-1], x[:, -1]

new_trainX = []



for elt in trainX:
    column_number = values.shape[1]-1
    elt_arranged = []
    for col_num in range(column_number):
        elt_arranged.append([elt[index] for index in [0+(col_num*column_number), 1+(col_num*column_number), 2+(col_num*column_number)]])
    elt_arranged = np.array(elt_arranged)
    new_trainX.append(elt_arranged.flatten())
new_trainX = np.array(new_trainX)


multiple_Y = []
counter_for_first_element = 0
for elt in new_trainX:
    if(counter_for_first_element>3):
        next_y = elt[0:3]
        multiple_Y.append(next_y)
    else:
        counter_for_first_element += 1
multiple_Y = np.array(multiple_Y)

def forecast_algorithm(model, future_day_amount):
    row = trainX[len(trainX)-1]
    forecasted_values = []
    x_row = asarray([x[len(x)-1]])[0]

    if(model == "rfr"):
        model = RandomForestRegressor(n_estimators=100, max_depth=6 , random_state=6)
        clf = MultiOutputRegressor(model).fit(x[:-4, :], multiple_Y)
        model.fit(trainX, trainy)

        for i in range(future_day_amount):
            multi_output = clf.predict([x_row])
            yhat = model.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))

        '''
        error_model = RandomForestRegressor(n_estimators=100, max_depth=6)
        clf_error = MultiOutputRegressor(error_model).fit(x[:int(len(x) * 0.80) - 4, :], multiple_Y[0:int(len(multiple_Y) * 0.8)])
        error_model.fit(x[0:int(len(x) * 0.8)], multiple_Y[0:int(len(multiple_Y) * 0.8)])
        preds = []
        
        for i in range(int(len(x) * 0.2)):
            multi_output = clf_error.predict(x[int(len(x) * 0.8)]:, x[int(len(x) * 0.8)]+4)
            yhat = error_model.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))
        '''

    # decision tree
    if (model == "dtr"):
        dt_reg = DecisionTreeRegressor(max_depth=22, max_features="auto", random_state=10)
        clf = MultiOutputRegressor(dt_reg).fit(x[:-4, :], multiple_Y)
        dt_reg.fit(trainX, trainy)

        for i in range(future_day_amount):
            multi_output = clf.predict([x_row])
            yhat = dt_reg.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))

    if(model =="xgb"):
        regressor = xgb.XGBRegressor(
                    n_estimators=100,
                    reg_lambda=1,
                    gamma=0,
                    max_depth=3
                )
        clf = MultiOutputRegressor(regressor).fit(x[:-4, :], multiple_Y)
        regressor.fit(trainX, trainy)

        for i in range(future_day_amount):
            multi_output = clf.predict([x_row])
            yhat = regressor.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))

    if (model == "svr"):
        svr_rbf = SVR(kernel='poly', C=1e2, gamma="scale")
        clf = MultiOutputRegressor(svr_rbf).fit(x[:-4, :], multiple_Y)
        svr_rbf.fit(trainX, trainy)

        for i in range(future_day_amount):
            multi_output = clf.predict([x_row])
            yhat = svr_rbf.predict(asarray([row]))

            row = row[4:]
            row = np.append(row, yhat)
            row = np.append(row, multi_output[0])

            x_row = x_row[4:]
            x_row = np.append(x_row, multi_output)
            x_row = np.append(x_row, yhat)

            forecasted_values.append(float(yhat))
    '''
    elif (model == "var"):
        model = VAR(data)
        num_columns = len(data.columns)
        results = model.fit(maxlags=num_columns, ic='aic')
        lag_order = results.k_ar
        y_preds = results.forecast(data.values[-lag_order:], future_day_amount)
        for element in y_preds:
            forecasted_values.append(element[-1])

    elif (model == "sarima"):
        index_count = 0
        new_df = data.copy()
        for column in data.columns:
            stepwise_fit = auto_arima(data[column], start_p=1, start_q=1,
                                      max_p=1, max_q=1, m=12,
                                      start_P=0, seasonal=True,
                                      d=None, D=1, trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)

            best_params = stepwise_fit.get_params()

            model = sm.tsa.statespace.SARIMAX(data[column],
                                              order=best_params["order"],
                                              seasonal_order=best_params["seasonal_order"])
            result = model.fit()

            forecast = result.predict(start=len(data),
                                      end=(len(data) - 1) + future_day_amount,
                                      typ='levels').rename('Forecast')
            forecasted_values.append(float(forecast))

    '''



    '''
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
    '''

    return forecasted_values


forecasted_values = forecast_algorithm("rfr", 50)
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
datelist = data.index.tolist() + dt[1]

for i in range(0, len(past_and_forecasted_values)):
    if i < len(past_values):
        col.append('blue')
    else:
        col.append('r')

graphdata = []

for i in range(len(past_and_forecasted_values)):
    plt.scatter(datelist[i], past_and_forecasted_values[i], c=col[i], s=10,
                linewidth=0)
    graphdata.append((datelist[i], past_and_forecasted_values[i]))

print("graph data = ", graphdata)


def calculateRMSE(actual_values, predicted_values):
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    return rmse


plt.ylabel("Mean Temperatures")
plt.xlabel("Time")
plt.title("Mean Temperature Forecasting")
ax = plt.gca()
ax.set_xlim([dt[0] - pd.DateOffset(5), dt[1][-1] + pd.DateOffset(5)])
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
