import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from xgboost import XGBRegressor

import matplotlib.pyplot as plt

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
df.head()
df.info()

df['date_time'] = pd.to_datetime(df['date_time'])

df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.day
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year
df['dayofweek'] = df['date_time'].dt.dayofweek

df = pd.get_dummies(
    df,
    columns=[
        'holiday',
        'weather_main',
        'weather_description'
    ],
    drop_first=True
)

X = df.drop(
    ['traffic_volume', 'date_time'],
    axis=1
)

y = df['traffic_volume']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


ada = AdaBoostRegressor(
    n_estimators=100,
    random_state=42
)

ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)


xgb = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)


def evaluate(y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    r2 = r2_score(y_true, pred)

    return mae, rmse, r2

rf_results = evaluate(y_test, rf_pred)
ada_results = evaluate(y_test, ada_pred)
xgb_results = evaluate(y_test, xgb_pred)

print("\nRandom Forest")
print(rf_results)

print("\nAdaBoost")
print(ada_results)

print("\nXGBoost")
print(xgb_results)

