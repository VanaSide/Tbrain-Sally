#%%
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

def merge_data(folder,output_file, file_pattern='*.csv'):
    csv_files = glob.glob(os.path.join(folder,file_pattern))
    if len(csv_files) != 17:
        print(f"警告：找到 {len(csv_files)} 個 CSV 檔案，預期需要 17 個。請檢查資料夾。")
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"合併完成，檔案儲存為{output_file}")

def transform_data(data):
    '''日照 Sunlight(Lux)'''
    data_sun = data[data['Sunlight(Lux)'] < 117758.2]
    X = data_sun[['Sunlight(Lux)']]
    y = data_sun[['Power(mW)']]
    model = LinearRegression()
    model.fit(X, y)
    # 預測發電量對應的光照度 (回推光照度)
    def predict_lux(power_value, model):
        # 根據回歸方程反推光照度
        return (power_value - model.intercept_[0]) / (model.coef_[0][0])
    data.loc[data['Sunlight(Lux)'] >= 117758.2, 'Sunlight(Lux)'] = data.loc[data['Sunlight(Lux)'] >= 117758.2, 'Power(mW)'].apply(lambda x: predict_lux(x, model))
    '''Temperature(°C) : 適當區間為10-35°C'''
    data.loc[data['Temperature(°C)'] < 10, 'Temperature(°C)'] = 10
    data.loc[data['Temperature(°C)'] > 35, 'Temperature(°C)'] = 35
    '''Humidity(%) : 適當區間 20-100% '''
    data.loc[data['Humidity(%)'] < 20, 'Humidity(%)'] = 20
    data.loc[data['Humidity(%)'] > 100, 'Humidity(%)'] = 100
    '''Pressure(hpa) : 大於1013.25不合理'''
    data.loc[data['Pressure(hpa)'] > 1013.25, 'Pressure(hpa)'] = 1013.25
    '''WindSpeed(m/s) : 不參考'''
    data.drop(columns=['WindSpeed(m/s)'], inplace=True)
    '''時間 : 1. 使用 Grouper 按每 10 分鐘進行分組，並對其他欄位進行平均
    2. 取 9:00 - 16:59 的資料
    3. 新增季、月、日、時間段、時間欄位'''
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data_group = data.set_index('DateTime').groupby(
    [pd.Grouper(freq='10min'), 'LocationCode']
    ).mean().reset_index()
    data_filtered = data_group[(data_group['DateTime'].dt.hour>=9) & (data_group['DateTime'].dt.hour<17)]
    data_filtered = data_filtered.copy()
    data_filtered.loc[:, 'Quarter'] = data_filtered['DateTime'].dt.quarter
    data_filtered.loc[:, 'Month'] = data_filtered['DateTime'].dt.month
    data_filtered.loc[:, 'Day'] = data_filtered['DateTime'].dt.day
    def time_of_day(hour):
        if 9 <= hour < 12:
            return 1
        elif 12 <= hour < 15:
            return 2
        else:
            return 3
    # 新增時間段欄位 : 假設白天、下午、傍晚這三個時段對太陽能發電有顯著的不同。
    data_filtered.loc[:, 'TimeOfDay'] = data_filtered['DateTime'].dt.hour.apply(time_of_day)
    data_filtered.loc[:, 'Hour'] = data_filtered['DateTime'].dt.hour
    data_filtered.loc[:, 'DateTimeString'] = data_filtered['DateTime'].dt.strftime('%Y%m%d%H%M')
    return data_filtered

def display_scores(scores):
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())
#%%
# merge_data('TrainingData','merged_data.csv')
# data = pd.read_csv('merged_data.csv')
data = pd.read_csv('TrainingData/L7_Train.csv')
# 根據地點分層抽樣切分訓練集與測試集
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
for train_index, test_index in split.split(data, data["LocationCode"]):
  train_set = data.loc[train_index]
  test_set = data.loc[test_index]
train_set = transform_data(train_set) 
test_set = transform_data(test_set) 
train_set_X = train_set.drop(['DateTime','DateTimeString',"Power(mW)"],axis=1)
train_set_Y = train_set["Power(mW)"].copy()
# 建立模型
param_grid = [{'criterion' : ['absolute_error']},]
tree_reg = DecisionTreeRegressor()
grid_search = GridSearchCV(tree_reg, param_grid, cv=10)
grid_search.fit(train_set_X, train_set_Y)
final_model = grid_search.best_estimator_
test_set_X = test_set.drop(['DateTime','DateTimeString',"Power(mW)"],axis=1)
test_set_Y = test_set["Power(mW)"].copy()
final_prediction = final_model.predict(test_set_X)
final_ae = mean_absolute_error(test_set_Y, final_prediction)
print(final_ae)
'''最佳超參數 & 最佳模型 : 在伺服器上跑
1. 定義模型及其參數網格
models = {
    'LinearRegression': (LinearRegression(), {}),
    'DecisionTree': (DecisionTreeRegressor(random_state=123), {
        'criterion' : ['squared_error', 'absolute_error'],
        'max_depth' : [3, 10, 30, None],
    }),
    'RandomForest': (RandomForestRegressor(random_state=123), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }),
    'SVM': (SVR(), {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    })
}
2. 定義存儲結果的字典
best_model = None
best_mae = float('inf')
best_model_name = ''
model_results = {}
3.對每個模型進行 GridSearchCV 超參數調優並比較 MAE
for model_name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(train_set_X, train_set_Y)
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(test_set_X)
    mae = mean_absolute_error(test_set_Y, y_pred)
    model_results[model_name] = mae

    if mae < best_mae:
        best_mae = mae
        best_model = best_estimator
        best_model_name = model_name

# 6. 打印結果
print(f"最佳模型: {best_model_name}, MAE: {best_mae}")
for model_name, mae in model_results.items():
    print(f"{model_name} MAE: {mae}")
'''
# %%
