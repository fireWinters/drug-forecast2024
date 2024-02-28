import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import xgboost
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from bayes_opt import BayesianOptimization


def scatter_plot_with_diagonal(y_true, y_pred,name):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, color='blue', label='Data Points', s=2)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--',
             label='45 Degree Line')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs. Predicted')
    plt.legend()
    plt.savefig(f'./{name}.png')
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
                # 增加一列日期，值为sheet_name
                date_str = f'2024-{sheet_name.replace(".", "-")}'  # 将点号替换为横杠，构造日期字符串
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

    return df


# Todo 药品类别


# 定义一个函数，用于调优XGBoost模型的超参数
def xgboost_cv_func(data, target, pbounds):
    
    # 定义内部函数，用于进行XGBoost模型的交叉验证
    def xgboost_crossval(max_depth, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
        # 将传入的超参数转换为字典形式
        params = {'max_depth': int(max_depth),
                  'n_estimators': int(n_estimators),
                  'gamma': gamma,
                  'min_child_weight': min_child_weight,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree,
                #   'tree_method': 'gpu_hist'  # 可选：使用GPU加速训练
                  }
        
        # 构建XGBoost回归模型
        model = xgb.XGBRegressor(**params)
        
        # 进行5折交叉验证，使用负均方误差作为评估指标，返回平均得分
        return cross_val_score(model, data, target, cv=5, scoring='neg_mean_squared_error').mean()

    # 使用贝叶斯优化器，传入交叉验证函数和超参数空间
    optimizer = BayesianOptimization(
        f=xgboost_crossval,
        pbounds=pbounds)

    # 最大化贝叶斯优化器，进行超参数优化
    optimizer.maximize(init_points=5, n_iter=50)
    
    # 获取优化后的最佳超参数配置
    best_params = optimizer.max
    best_params = best_params['params']
    
    # 设置最佳超参数的'tree_method'为'gpu_hist'，并将'max_depth'和'n_estimators'转换为整数类型
    best_params['tree_method'] = 'gpu_hist'
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    
    # 返回最佳的超参数配置
    return best_params

if __name__ == '__main__':
    current_directory = r'/Users/callustang/tangCode/shantouCode/drug-forecast2024'
    data_folder_path = os.path.join(current_directory, 'realData')

    df = get_all_dataframes()

    data = data_process(df, [1, 3, 7, 14, 30, 60, 90])
    data = data_cleaning(data)
    data = label_process(data, 14)

    data = data[~data['y'].isna()]  # 排除y的nan值
    data = data[data['y'] < data['y'].quantile(0.99)]

    X, y = data[['month', 'quarter', 'de_sum_1', 'de_mean_1', 'de_max_1', 'de_min_1', 'de_sum_3', 'de_mean_3', 'de_max_3',
         'de_min_3',
         'de_sum_7', 'de_mean_7', 'de_max_7', 'de_min_7', 'de_sum_14', 'de_mean_14', 'de_max_14', 'de_min_14',
         'de_sum_30', 'de_mean_30', 'de_max_30', 'de_min_30', 'de_sum_60', 'de_mean_60', 'de_max_60', 'de_min_60',
         'de_sum_90', 'de_mean_90', 'de_max_90', 'de_min_90', ]], data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    pbounds = {'max_depth': (3, 8),
               'gamma': (0, 1),
               'n_estimators': (100, 500),
               'min_child_weight': (0, 10),
               'subsample': (0.5, 1),
               'colsample_bytree': (0.5, 1)}

    best_params = xgboost_cv_func(X_train, y_train, pbounds)

    xgb_reg = xgboost.XGBRegressor(**best_params)
    xgb_reg = xgboost.XGBRegressor()
    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_test)
    y_train_pred = xgb_reg.predict(X_train)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'mes: {mse}, r2: {r2}')

    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)
    print(f'mes: {mse}, r2: {r2}')

    scatter_plot_with_diagonal(y_test, y_pred, 'test预测效果')
    scatter_plot_with_diagonal(y_train, y_train_pred, 'train预测效果')

