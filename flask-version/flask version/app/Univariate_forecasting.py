# finalize model and make a prediction for monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import xgboost as xgb
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# transform a time series dataset into a supervised learning dataset
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


# load the dataset
series = read_csv('C://Users//Administrator//Desktop//login//app//Electric_cost.csv', header=0, index_col=0)
values = series.values
# print(values)
# transform the time series data into supervised learning


train = series_to_supervised(values, n_in=8)

# split into input and output columns
trainX, trainy = train[:, :-1], train[:, -1]


def forecast_algorithm(model, future_day_amount):
    row = values[-8:].flatten()
    forecasted_values = []




    if(model =="xgb"):
        regressor = xgb.XGBRegressor(
                    n_estimators=100,
                    reg_lambda=1,
                    gamma=0,
                    max_depth=3
                )
        regressor.fit(trainX, trainy)

        for i in range(future_day_amount):
            yhat = regressor.predict(asarray([row]))
            # print('Input: %s, Predicted: %.3f' % (row, yhat[0]))
            row = np.delete(row, 0)
            row = np.append(row, yhat)
            forecasted_values.append(float(yhat))
#s

    #print("Results")
    #print(forecasted_values)
    return forecasted_values

forecasted_values = forecast_algorithm("xgb", 50)


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


def graphVal(name):
    a = name
    print(a)



"""plt.ylabel("Prices")
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
plt.show()"""
####### database işleri#######
# @app.route("/graph", methods=['GET', 'POST'])
# @login_required
# def profile():
#     form = GraphForm()
#     if form.validate_on_submit():
#         if form.days.data:
#             istenen_gun = form.days.data
#             forecast = RandomForestRegressor(istenen_gun)
#             pred = PredictedVal(prediction=forecast)
#             db.session.add(pred)
#     db.session.commit()
#
#
# class PredictedVal(db.Model):
#     predicted = db.Column(db.Integer)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#
#     def _repr_(self):
#         return f"PredictedVal('{self.predicted}')"




