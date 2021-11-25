import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import VAR
from pandas.plotting import register_matplotlib_converters
import json
import base64
import urllib

register_matplotlib_converters()
from pmdarima import auto_arima
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse

# In[2]:

import pandas as pd
all_tables = pd.read_html(
    "https://www.proff.no/regnskap/yara-international-asa/oslo/hovedkontortjenester/IGB6AV410NZ/"
)
with pd.ExcelWriter('output.xlsx') as writer:
    # Last 4 tables has the 'konsernregnskap' data
    for idx, df in enumerate(all_tables[4:8]):
        # Remove last column (empty)
        df = df.drop(df.columns[-1], axis=1)
        df.to_excel(writer, "Table {}".format(idx))

print(df)

df = pd.read_csv('DailyDelhiClimateTrain.csv',
                 index_col='date',
                 parse_dates=True)
df.head()


# Preprocessor

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


convert_to_datetime(df)
df.sort_values(by=['date'], inplace=True)
df = df._get_numeric_data()

if ((df.duplicated()).sum() > 0):
    print("There are:", (df.duplicated()).sum(), "duplicates.")
    df.drop_duplicates(inplace=True)

df = df.fillna(method='ffill').fillna(method='bfill')

# useful variables

df_columns_length = len(df.columns)
target_col_name = df.columns[df_columns_length - 1]

decompose_data = seasonal_decompose(df[target_col_name], model="additive")
decompose_data.plot();


# seasonality.plot()


# In[15]:


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
    return new_df


# In[74]:


def create_model_prediction(model_name):
    forecast_days = int(input("How many days you want to predict?: "))
    new_df = extend_dataset(forecast_days)

    columns = new_df.columns
    train_columns = columns[:-1]
    test_columns = columns[-1:]
    X = new_df[train_columns]
    Y = new_df[test_columns]

    x_train = X[:len(df)]
    x_test = X[len(df):]
    y_train = Y[:-forecast_days]
    y_test = Y[-forecast_days:]

    if (model_name == "linear"):

        x_train = sm.add_constant(x_train)

        results = sm.OLS(y_train, x_train).fit()

        # sonuclar.summary()
        x_test = sm.add_constant(x_test)

        y_preds = results.predict(x_test)
        return (y_preds, y_test)

    elif (model_name == "var"):
        model = VAR(new_df)
        num_columns = len(new_df.columns)
        results = model.fit(maxlags=num_columns, ic='aic')
        lag_order = results.k_ar
        y_preds = results.forecast(new_df.values[-lag_order:], len(x_test))
        y_preds = y_preds[:, -1]
        print(y_preds)
        return (y_preds, y_test)

    else:
        print("Not a model we have, try: var or linear")
        return


# In[75]:


model_type = input("Which model do you want to use?")
result_prediction, results = create_model_prediction(model_type)

# In[76]:


baslik_font = {'family': 'arial', 'color': 'darkred', 'weight': 'bold', 'size': 15}
eksen_font = {'family': 'arial', 'color': 'darkblue', 'weight': 'bold', 'size': 10}
plt.figure(dpi=100)

plt.scatter(results, result_prediction)
plt.plot(results, results, color="red")
plt.xlabel("Gerçek Değerler", fontdict=eksen_font)
plt.ylabel("Tahmin edilen Değerler", fontdict=eksen_font)
plt.title("Ücretler: Gerçek ve tahmin edilen değerler", fontdict=baslik_font)
plt.savefig("TSF.png", dpi=100)
plt.show()

TSF_data = {}
with open('TSF.png', mode='rb') as file:
    img = file.read()
TSF_data['img'] = base64.encodebytes(img).decode('utf-8')
print(json.dumps(TSF_data))
