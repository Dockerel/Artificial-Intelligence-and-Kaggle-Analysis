import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("train_bsd.csv")
test = pd.read_csv("test_bsd.csv")

print(train.shape)
print(test.shape)

datasets = [train, test]

submit = pd.read_csv("sampleSubmission.csv")
print(submit.shape)

for dataset in datasets:
    dataset["datetime"] = pd.to_datetime(dataset["datetime"])

for dataset in datasets:
    dataset["datetime-year"] = dataset["datetime"].dt.year
    dataset["datetime-hour"] = dataset["datetime"].dt.hour

for dataset in datasets:
    dataset["datetime-dayofweek"] = dataset["datetime"].dt.day_name()
    dataset[["datetime-dayofweek"]]

for dataset in datasets:
    dataset["datetime-dayofweek_Sun"] = dataset["datetime-dayofweek"] == "Sunday"
    dataset["datetime-dayofweek_Mon"] = dataset["datetime-dayofweek"] == "Monday"
    dataset["datetime-dayofweek_Tue"] = dataset["datetime-dayofweek"] == "Tuesday"
    dataset["datetime-dayofweek_Wed"] = dataset["datetime-dayofweek"] == "Wednesday"
    dataset["datetime-dayofweek_Thu"] = dataset["datetime-dayofweek"] == "Thursday"
    dataset["datetime-dayofweek_Fri"] = dataset["datetime-dayofweek"] == "Friday"
    dataset["datetime-dayofweek_Sat"] = dataset["datetime-dayofweek"] == "Saturday"


for dataset in datasets:
    dataset.loc[dataset["weather"] == 4, "weather"] = 3


feature_names = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "datetime-year",
    "datetime-hour",
    "datetime-dayofweek_Mon",
    "datetime-dayofweek_Tue",
    "datetime-dayofweek_Wed",
    "datetime-dayofweek_Thu",
    "datetime-dayofweek_Fri",
    "datetime-dayofweek_Sat",
    "datetime-dayofweek_Sun",
]


x = train[feature_names]
y_log_casual = np.log(train["casual"] + 1)
y_log_registered = np.log(train["registered"] + 1)
y = train["count"]
x_test = test[feature_names]


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=37)

# ## model validation
from sklearn.model_selection import cross_val_predict

y_predict_log_casual = cross_val_predict(model, x, y_log_casual, cv=20)
y_predict_log_registered = cross_val_predict(model, x, y_log_registered, cv=20)
y_predict_casual = np.exp(y_predict_log_casual) - 1
y_predict_registered = np.exp(y_predict_log_registered) - 1
y_predict = y_predict_casual + y_predict_registered

score_mae = abs(y - y_predict).mean()
score_mse = ((y - y_predict) ** 2).mean()
score_rmse = (score_mse) ** 0.5
print(
    f"score_mae = {score_mae:.1f}, score_mse = {score_mse:.1f}, score_rmse = {score_rmse:.1f}"
)

from sklearn.metrics import mean_absolute_error

score_mae1 = mean_absolute_error(y, y_predict)
from sklearn.metrics import mean_squared_error

score_mse1 = mean_squared_error(y, y_predict)
score_rmse1 = np.sqrt(score_mse1)
from sklearn.metrics import mean_squared_log_error

score_msle = mean_squared_log_error(y, y_predict)
score_rmsle = np.sqrt(score_msle)
print(
    f"score(MAE) = {score_mae1:.1f}, score(MSE) = {score_mse1:.1f}, score(RMSE) = {score_rmse1:.1f}, score(RMSLE) = {score_rmsle:.5f}"
)


# ## fit & predict & submit
model.fit(x, y_log_casual)
log_casual_predlist = model.predict(x_test)
model.fit(x, y_log_registered)
log_registered_predlist = model.predict(x_test)
casual_predlist = np.exp(log_casual_predlist) - 1
registered_predlist = np.exp(log_registered_predlist) - 1
predlist = casual_predlist + registered_predlist

submit["count"] = predlist
submit.to_csv("randfore.csv", index=False)
