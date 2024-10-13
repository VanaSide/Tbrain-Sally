import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

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
    2. 取 9:00 - 16:59 的資料'''
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data_group = data.set_index('DateTime').groupby(
    [pd.Grouper(freq='10T'), 'LocationCode']
    ).mean().reset_index()
    data_filtered = data_group[(data_group['DateTime'].dt.hour>=9) & (data_group['DateTime'].dt.hour<17)]
    return data_filtered

def display_scores(scores):
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())

merge_data('TrainingData','merged_data.csv')
data = pd.read_csv('merged_data.csv')
# 根據地點分層抽樣切分訓練集與測試集
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
for train_index, test_index in split.split(data, data["LocationCode"]):
  train_set = data.loc[train_index]
  test_set = data.loc[test_index]
train_set = transform_data(train_set) 
test_set = transform_data(test_set) 
train_set_X = train_set.drop("Power(mW)",axis=1)
train_set_Y = train_set["Power(mW)"].copy()
# 建立模型
tree_reg = DecisionTreeRegressor()
scores = cross_val_score(tree_reg, train_set_X, train_set_Y,
                         scoring='neg_mean_squared_error', cv = 10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores) 
joblib.dump(tree_reg, "tree.pkl")