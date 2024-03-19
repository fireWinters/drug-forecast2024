import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import xgboost
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima_model import ARIMA  # 用于使用ARIMA模型
import itertools  # 用于创建迭代器


def scatter_plot_with_diagonal(y_true, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, color='blue', label='Data Points', s=2)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--',
             label='45 Degree Line')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs. Predicted')
    plt.legend()
    plt.show()


def get_all_dataframes():
    """获取2020原始数据下excel，并将每一个sheet拼接成一个df"""
    all_dataframes = []
    # 遍历文件夹中的每个xlsx文件
    for filename in tqdm(os.listdir(data_folder_path)):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(data_folder_path, filename)

            # 读取xlsx文件的所有sheet
            sheets_dict = pd.read_excel(file_path, sheet_name=None)

            # 遍历每个sheet
            for sheet_name, sheet_data in sheets_dict.items():
                 # 如果sheet_name是XXXX.X.XX格式，就用sheet_name，否则用文件名
                if sheet_name.count('.') == 2:
                    # 把sheet_name中的点号替换为横杠
                    date_str = sheet_name.replace(".", "-")
                    # date_str = sheet_name
                else:
                    date_str = filename.split('.')[0] + '-' + sheet_name.replace(".", "-")
                # 增加一列日期，值为sheet_name
                # date_str = f'2020-{sheet_name.replace(".", "-")}'  # 将点号替换为横杠，构造日期字符串
                sheet_data['日期'] = datetime.strptime(date_str, '%Y-%m-%d').date()
                # 添加到all_dataframes列表中
                all_dataframes.append(sheet_data)

    # 将所有数据拼接成一个DataFrame
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)

    final_dataframe = final_dataframe.drop(['入库单位', '基本单位'], axis=1)
    return final_dataframe


def data_cleaning(df):
    """把前九十天没有减少数量的剔除掉"""

    df = df[df['de_sum_90'] > 0]
    # 把药品分类代码列值为删的数据全部剔除
    df = df[df['药品分类代码'] != '删']
    # 删除药品分类代码为空的数据
    df = df[~df['药品分类代码'].isna()]
    return df


def label_process(df, days=7):
    df['y'] = df.groupby('药品名称')['减少数量'].transform(
        lambda x: x.rolling(window=days, min_periods=1).sum().shift(-7))

    return df


def data_process(df, day_lst):
    """
    处理
    1.日期
    2.季度
    3.时序
    特征
    """
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(by=['药品名称', '日期'])

    df['month'] = df['日期'].dt.month
    df['quarter'] = (df['month'] - 1) // 3 + 1

    df['减少数量'] = df['减少数量'].abs()

    for days in tqdm(day_lst):
        df[f'de_sum_{days}'] = df.groupby('药品名称')['减少数量'].transform(
            lambda x: x.rolling(window=days, min_periods=1).sum().shift(1))

        df[f'de_mean_{days}'] = df.groupby('药品名称')['减少数量'].transform(
            lambda x: x.rolling(window=days, min_periods=1).mean().shift(1))

        df[f'de_max_{days}'] = df.groupby('药品名称')['减少数量'].transform(
            lambda x: x.rolling(window=days, min_periods=1).max().shift(1))

        df[f'de_min_{days}'] = df.groupby('药品名称')['减少数量'].transform(
            lambda x: x.rolling(window=days, min_periods=1).min().shift(1))

    for days in tqdm(day_lst):
        df[f'bothcode_de_sum_{days}'] = df.groupby('药品分类代码')['减少数量'].transform(
            lambda x: x.rolling(window=days, min_periods=1).sum().shift(1))

        df[f'bothcode_de_mean_{days}'] = df.groupby('药品分类代码')['减少数量'].transform(
            lambda x: x.rolling(window=days, min_periods=1).mean().shift(1))

        df[f'bothcode_de_max_{days}'] = df.groupby('药品分类代码')['减少数量'].transform(
            lambda x: x.rolling(window=days, min_periods=1).max().shift(1))

        df[f'bothcode_de_min_{days}'] = df.groupby('药品分类代码')['减少数量'].transform(
            lambda x: x.rolling(window=days, min_periods=1).min().shift(1))

    return df


def xgboost_cv_func(data, target, pbounds):
    def xgboost_crossval(max_depth, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
        params = {'max_depth': int(max_depth),
                  'n_estimators': int(n_estimators),
                  'gamma': gamma,
                  'min_child_weight': min_child_weight,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree,
                #   'tree_method': 'gpu_hist'
                  }
        model = xgb.XGBRegressor(**params)
        return cross_val_score(model, data, target, cv=5, scoring='neg_mean_squared_error').mean()

    optimizer = BayesianOptimization(
        f=xgboost_crossval,
        pbounds=pbounds)

    optimizer.maximize(init_points=5, n_iter=50)
    best_params = optimizer.max
    best_params = best_params['params']
    best_params['tree_method'] = 'gpu_hist'
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    return best_params


def lightgbm_cv_func(data, target, pbounds):
    def lgb_crossval(max_depth, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
        params = {'max_depth': int(max_depth),
                  'n_estimators': int(n_estimators),
                  'gamma': gamma,
                  'min_child_weight': min_child_weight,
                  'subsample': subsample}
        model = lgb.LGBMRegressor(**params)
        return cross_val_score(model, data, target, cv=5, scoring='neg_mean_squared_error').mean()

    optimizer = BayesianOptimization(
        f=lgb_crossval,
        pbounds=pbounds)

    optimizer.maximize(init_points=5, n_iter=50)
    best_params = optimizer.max
    best_params = best_params['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    return best_params


def model_train(X_train, X_test, y_train, y_test, pbounds, model_type):
    if model_type == 'xgboost':
        best_params = xgboost_cv_func(X_train, y_train, pbounds)

        model = xgboost.XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
    elif model_type == 'lightgbm':
        # 由于lightgbm的参数中没有gamma，所以需要把gamma从pbounds中pop出来
        pbounds1 = pbounds.copy()
        pbounds1.pop('gamma')
        best_params = lightgbm_cv_func(X_train, y_train, pbounds1)

        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

    return {'model_type': model_type, 'model': model,
            'test_mse': test_mse, 'test_r2': test_r2,
            'test_mae': test_mae, 'test_mape': test_mape,
            'train_mse': train_mse, 'train_r2': train_r2,
            'train_mae': train_mae, 'train_mape': train_mape}


def plot_prediction(date, y_true, y_pred, model_type,drug_code,drug_name,save_name=None):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    drug_code = drug_code
    drug_name = drug_name

    data = pd.DataFrame({
        'date': pd.to_datetime(date),
        'y_true': y_true,
        'y_pred': y_pred,
    })

    # 对日期进行排序
    data = data.sort_values('date')

    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['y_true'], label='True')
    plt.plot(data['date'], data['y_pred'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(f'moduleType:{model_type},code:{drug_code},r2: {r2:.2f}, mae: {mae:.2f}, mape: {mape:.2f}, mse: {mse:.2f},name:{drug_name}')
    plt.legend()
    plt.xticks(data['date'], rotation=45)
    plt.tight_layout()
    # 保存图片，用模型名称和药品分类代码命名
    if save_name:
        plt.savefig(f'./{save_name}.png')
    else:
        plt.savefig(f'./{model_type}_{drug_code}.png')
    # plt.show()

# 定义一个函数，用于评估ARIMA模型
def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) - 50)
    train, test = X[0:train_size], X[train_size:]
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    forecast = model_fit.predict(1,len(test))
    error = mean_squared_error(test, forecast)
    return error

# 定义一个函数，用于评估多个模型
def evaluate_models(dataset, pdq):
    best_score, best_cfg = float("inf"), None
    for i in pdq:

        mse = evaluate_arima_model(dataset, i)
        if mse < best_score:
            best_score, best_cfg = mse, i

    print(' - Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
# 定义LSTM模型
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :-1]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)


def create_lstm_dataset(dataset, x_cols, y_col, look_back=10):
    datasX, datasY = np.empty((0, look_back, len(x_cols))), np.empty((0, ))  # 使用 np.empty() 创建空数组
    drugs = dataset['药品名称'].unique().tolist()
    for j, i in tqdm(enumerate(drugs)):
        dataset_1 = dataset[dataset['药品名称'] == i][x_cols + [y_col]].values
        x, y = create_dataset(dataset_1, look_back=look_back)
        if j == 0:
            datasX = x
            datasY = y
        else:

            if x.shape[0] == 0:
                print(datasX.shape, x.shape)
                continue
            else:
                datasX = np.concatenate((datasX, x), axis=0)
                datasY = np.concatenate((datasY, y), axis=0)
    return datasX, datasY


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_lstm(model, criterion, optimizer, train_loader, num_epochs):
    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float()
            labels = labels.float()
            outputs = model(inputs)
            # print(labels, outputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                         loss.item()))

if __name__ == '__main__':
    current_directory = r'/Users/callustang/tangCode/shantouCode/drug-forecast2024'
    data_folder_path = os.path.join(current_directory, 'realData')
    df = get_all_dataframes()

    cate_data = pd.read_csv(os.path.join(current_directory, 'drug_with_category_2024.csv'))
    df = pd.merge(df, cate_data[['药品名称', '药品分类代码']], how='left', on='药品名称')

    data = data_process(df, [1, 3, 7, 14, 30, 60, 90, 180])
    data = data_cleaning(data)
    # 数据输出为csv格式
    data.to_csv('./category_data_all_20240319.csv', index=False)

    data = label_process(data, 7)

    data = data[~data['y'].isna()]  # 排除y的nan值
    data = data[data['y'] < data['y'].quantile(0.99)]  # 排除99分位数以上的值

    del df
    ### 对单独一个药品做模型

    x_cols = ['month', 'quarter', 'de_sum_1', 'de_mean_1', 'de_max_1', 'de_min_1', 'de_sum_3', 'de_mean_3', 'de_max_3',
              'de_min_3', 'de_sum_7', 'de_mean_7', 'de_max_7', 'de_min_7', 'de_sum_14', 'de_mean_14', 'de_max_14',
              'de_min_14',
              'de_sum_30', 'de_mean_30', 'de_max_30', 'de_min_30', 'de_sum_60', 'de_mean_60', 'de_max_60', 'de_min_60',
              'de_sum_90', 'de_mean_90', 'de_max_90', 'de_min_90', 'bothcode_de_sum_1', 'bothcode_de_mean_1', 'bothcode_de_max_1',
              'bothcode_de_min_1', 'bothcode_de_sum_3', 'bothcode_de_mean_3', 'bothcode_de_max_3', 'bothcode_de_min_3',
              'bothcode_de_sum_7', 'bothcode_de_mean_7', 'bothcode_de_max_7', 'bothcode_de_min_7', 'bothcode_de_sum_14',
              'bothcode_de_mean_14', 'bothcode_de_max_14', 'bothcode_de_min_14', 'bothcode_de_sum_30', 'bothcode_de_mean_30',
              'bothcode_de_max_30', 'bothcode_de_min_30', 'bothcode_de_sum_60', 'bothcode_de_mean_60', 'bothcode_de_max_60',
              'bothcode_de_min_60', 'bothcode_de_sum_90', 'bothcode_de_mean_90', 'bothcode_de_max_90', 'bothcode_de_min_90']

    X, y = data[x_cols], data['y']

    # out of time sample
    # oot_index = data[(data['日期'] >= '2020-12-01')].index
    pre_date='2023-05-01'
    oot_index = data[(data['日期'] >= pre_date)].index
    oot_x, oot_y = X[X.index.isin(oot_index)], y[oot_index]

    drug_name = data.loc[oot_index, '药品名称'].values[0]
    drug_code = data.loc[oot_index, '药品分类代码'].values[0]

    # train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    pbounds = {'max_depth': (3, 8),
               'gamma': (0, 1),
               'n_estimators': (100, 500),
               'min_child_weight': (0, 10),
               'subsample': (0.5, 1),
               'colsample_bytree': (0.5, 1)}

    # training models
    model_traineds = []
    model_types = ['xgboost', 'lightgbm']
    for model_type in model_types:
        model_trained = model_train(X_train, X_test, y_train, y_test, pbounds, model_type)
        model_traineds.append(model_trained)

    # evaluation models
    for model_trained in model_traineds:
        model = model_trained['model']
        oot_pred = model.predict(oot_x)
        date = data.loc[oot_index, '日期']
        plot_prediction(date, oot_y, oot_pred, model_type,drug_code,drug_name, save_name=None)


    # lstm
    ## 1.用0来填补缺失值，lstm模型不接受缺失值
    data[x_cols] = data[x_cols].fillna(0)
    ## 2.标准化处理，量纲对lstm模型影响巨大，要统一处理
    scaler_x = MinMaxScaler()
    scaler_y = StandardScaler()

    data[x_cols] = scaler_x.fit_transform(data[x_cols].values)
    data['y'] = scaler_y.fit_transform(data['y'].values.reshape(-1, 1)) # y值是否要标准化，需要考虑，MinMaxScaler来处理y值，效果不好

    lstm_x, lstm_y = create_lstm_dataset(data, x_cols, 'y')

    input_size = len(x_cols)
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 5
    learning_rate = 0.01

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(torch.tensor(lstm_x, dtype=torch.float32), torch.tensor(lstm_y, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    train_lstm(model, criterion, optimizer, train_loader,  num_epochs)

    # 验证模型效果
    model.eval() # 设置模型为评估模式
    predictions = []
    targets = []
    for batch_x, batch_y in tqdm(train_loader):
        with torch.no_grad():
            outputs = model(batch_x)

        predictions.extend(outputs.cpu().numpy())
        targets.extend(batch_y.cpu().numpy())

    r2 = r2_score(scaler_y.inverse_transform(np.array(targets).reshape(-1, 1)), scaler_y.inverse_transform(predictions))
    mse = mean_squared_error(scaler_y.inverse_transform(np.array(targets).reshape(-1, 1)), scaler_y.inverse_transform(predictions))
    mse_ = mean_squared_error(targets, predictions)

    # scatter_plot_with_diagonal(y_test, y_pred)
    # scatter_plot_with_diagonal(y_train, y_train_pred)

    # model_traineds[0]['model']
