#%%
import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ValidationCurveDisplay
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

#%%
def merge_data(folder,output_file, file_pattern='*.csv'):
    csv_files = glob.glob(os.path.join(folder,file_pattern))
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"合併完成，檔案儲存為{output_file}")

def transform_data(data):
    # 日照 Sunlight(Lux)
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
    # Temperature(°C) : 適當區間為10-35°C
    data.loc[data['Temperature(°C)'] < 10, 'Temperature(°C)'] = 10
    data.loc[data['Temperature(°C)'] > 35, 'Temperature(°C)'] = 35
    # Humidity(%) : 適當區間 20-100% 
    data.loc[data['Humidity(%)'] < 20, 'Humidity(%)'] = 20
    data.loc[data['Humidity(%)'] > 100, 'Humidity(%)'] = 100
    # Pressure(hpa) : 大於1013.25不合理
    data.loc[data['Pressure(hpa)'] > 1013.25, 'Pressure(hpa)'] = 1013.25
    # WindSpeed(m/s) : 不參考
    # data.drop(columns=['WindSpeed(m/s)'], inplace=True)
    ''' 時間 : 1. 使用 Grouper 按每 10 分鐘進行分組，並對其他欄位進行平均
    2. 新增季、月、日、時間段、時間欄位'''
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data_group = data.set_index('DateTime').groupby(
    [pd.Grouper(freq='10min'), 'LocationCode']
    ).mean().reset_index()
    '''
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
    data_filtered.loc[:, 'Hour'] = data_filtered['DateTime'].dt.hour'''
    data_group.loc[:, 'DateTimeString'] = data_group['DateTime'].dt.strftime('%Y%m%d%H%M')
    data_group.loc[:, 'LocationCode'] = data_group['LocationCode'].apply(lambda x : ('0'+ str(x)) if x < 10 else str(x))
    data_group.loc[:, 'Serial'] = data_group['DateTimeString'] + data_group['LocationCode']
    return data_group

def display_scores(scores):
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())

#%%
'''載入訓練資料
merge_data('TrainingData','merged_data.csv')'''
# DataName = os.getcwd()+'\ExampleTrainData(AVG)\AvgDATA_07.csv'
# data = pd.read_csv(DataName, encoding='utf-8')
merge_data('TrainingData','merged_data_Additional_V2.csv')
DataName = os.getcwd()+'\merged_data_Additional_V2.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
SourceData = transform_data(SourceData)
SourceData_AVG = SourceData[(SourceData ['DateTime'].dt.hour>=9) & (SourceData['DateTime'].dt.hour<17)]
SourceData_IncompleteAVG = SourceData[(SourceData ['DateTime'].dt.hour<9)]

#迴歸分析 選擇要留下來的資料欄位
#(風速,大氣壓力,溫度,濕度,光照度)
#(發電量)
Regression_X_train = SourceData_AVG[['WindSpeed(m/s)','Pressure(hpa)','Temperature(°C)','Humidity(%)','Sunlight(Lux)']].values
Regression_y_train = SourceData_AVG[['Power(mW)']].values

#LSTM 選擇要留下來的資料欄位
#(風速,大氣壓力,溫度,濕度,光照度)
AllOutPut = SourceData_AVG[['WindSpeed(m/s)','Pressure(hpa)','Temperature(°C)','Humidity(%)','Sunlight(Lux)']].values

#正規化
LSTM_MinMaxModel = MinMaxScaler().fit(AllOutPut)
AllOutPut_MinMax = LSTM_MinMaxModel.transform(AllOutPut)

X_train = []
y_train = []
# 設定LSTM往前看的筆數和預測筆數
LookBackNum = 12 # LSTM往前看的筆數
ForecastNum = 48 # 預測筆數
for i in range(LookBackNum, len(AllOutPut_MinMax)):
    X_train.append(AllOutPut_MinMax[i-LookBackNum:i, :])
    y_train.append(AllOutPut_MinMax[i, :])
X_train = np.array(X_train)
y_train = np.array(y_train)
#(samples 是訓練樣本數量,timesteps 是每個樣本的時間步長,features 是每個時間步的特徵數量)
X_train = np.reshape(X_train,(X_train.shape [0], X_train.shape [1], 5))

#%%
'''LSTM Model
'''
regressor = Sequential()
regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressor.add(LSTM(units = 64))
regressor.add(Dropout(0.2))
# output layer
regressor.add(Dense(units = 5))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#開始訓練
regressor.fit(X_train, y_train, epochs = 10, batch_size = 128)
#保存模型
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
joblib.dump(regressor, 'WheatherLSTM_'+NowDateTime+'.h5')
#%%
'''Regression Model
1. 透過ValidationCurveDisplay判斷超參數預測比較好的區間
ValidationCurveDisplay.from_estimator(
    tree_reg, train_set_X, train_set_Y, param_name = 'max_depth', param_range = np.arange(1,100,10)
)
2. 透過GridSearchCV選出最佳超參數
3. 模型擬合
4. 測試集資料 X, Y
5. 模型預測
# 印出GridSearchCV調參過程
means = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
for mean, params in zip(means,params):
    print("%f with: %r " % (mean,params))
'''
RegressionModel = DecisionTreeRegressor()
param_grid = [{'criterion' :  ['absolute_error'],
               'max_depth' : [3,10,30]}]
grid_search = GridSearchCV(RegressionModel, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(LSTM_MinMaxModel.transform(Regression_X_train), Regression_y_train)
final_reg_model = grid_search.best_estimator_
# 儲存模型
joblib.dump(final_reg_model, 'WheatherRegression_'+NowDateTime)

#%%
'''預測數據'''
#載入模型
regressor = joblib.load('WheatherLSTM_2024-11-25T16_54_10Z.h5')
Regression = joblib.load('WheatherRegression_2024-11-18T17_53_04Z')

DataName = os.getcwd()+r'\TestData\upload(no answer).csv'
test_set = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = test_set[target].values

inputs = [] #存放參考資料
PredictOutput = [] #存放預測值(天氣參數)
PredictPower = [] #存放預測值(發電量) 

count = 0
while(count < len(EXquestion)):
  print('count : ',count)
  LocationCode = int(EXquestion[count])
  strLocationCode = str(LocationCode)[-2:]
  if LocationCode < 10 :
    strLocationCode = '0'+LocationCode

  ReferTitle = SourceData_IncompleteAVG[['Serial']].values
  ReferData = SourceData_IncompleteAVG[['WindSpeed(m/s)','Pressure(hpa)','Temperature(°C)','Humidity(%)','Sunlight(Lux)']].values
  
  inputs = []#重置存放參考資料

  #找到相同的一天，把12個資料都加進inputs
  for DaysCount in range(len(ReferTitle)):
    if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]):
      TempData = ReferData[DaysCount].reshape(1,-1)
      inputs.append(TempData)

  #用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
  for i in range(ForecastNum) :

    #print(i)
    
    #將新的預測值加入參考資料(用自己的預測值往前看)
    if i > 0 :
      inputs.append(PredictOutput[i-1].reshape(1,5))

    #切出新的參考資料12筆(往前看12筆)
    X_test = []
    X_test.append(inputs[0+i:LookBackNum+i])
    
    #Reshaping
    NewTest = np.array(X_test)
    NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 5))
    
    predicted = regressor.predict(NewTest)
    PredictOutput.append(predicted)
    PredictPower.append(np.round(Regression.predict(predicted),2).flatten())
  
  #每次預測都要預測48個，因此加48個會切到下一天
  #0~47,48~95,96~143...
  count += 48

#寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame([EXquestion,PredictPower], columns=['序號','答案'])

# 將 DataFrame 寫入 CSV 檔案
df.to_csv('output_' + NowDateTime + '.csv', index=False, encoding='utf-8', line_terminator='\n') 
print('Output CSV File Saved')

'''
final_ae = mean_absolute_error(test_set_Y, final_prediction)
model_results = pd.DataFrame(columns=['model_name & best_params','mae'])
model_results.loc[1] = [final_model,final_ae]
print(model_results)
'''

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
actual = test_set['答案'].values
predict = df['答案'].values
ae = sum(abs(predict-actual))