import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
# import xgboost
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 这个函数用于绘制散点图，并在图中加入一条45度对角线，方便比较真实值和预测值之间的关系。
def scatter_plot_with_diagonal(y_true, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, color='blue', label='Data Points', s=2)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='45 Degree Line')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs. Predicted')
    plt.legend()
    plt.show()

# 这个函数用于获取指定文件夹中的所有xlsx文件，并将每个文件中的每个sheet拼接成一个DataFrame。
def get_all_dataframes():
    """获取原始数据下excel，并将每一个sheet拼接成一个df"""
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

# 这个函数用于数据清洗，剔除前九十天没有减少数量的数据。
def data_cleaning(df):
    """把前九十天没有减少数量的剔除掉"""

    df = df[df['de_sum_90']>0]
    return df

# 这个函数用于处理标签数据，计算每个药品名称在指定天数内的减少数量。
def label_process(df, days=7):

    df[f'y_{days}'] = df.groupby('药品名称')['减少数量'].transform(
        lambda x: x.rolling(window=days, min_periods=1).sum().shift(-7))

    return df

# 这个函数用于处理数据特征，包括日期、季度、时序特征，并计算每个药品名称在不同天数内的减少数量的统计特征。
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

    for days in day_lst:

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


# 首先指定了数据文件夹路径，然后调用get_all_dataframes函数获取所有数据，并进行数据处理和特征工程。接着使用XGBoost模型训练数据，并计算模型的均方误差（MSE）和决定系数（R^2）。最后绘制了真实值和预测值的散点图。
if __name__ == '__main__':
    current_directory = r'C:\Users\Zz\Desktop\project\callusTang'
    data_folder_path = os.path.join(current_directory, '2020原始数据')

    df = get_all_dataframes()

    data = data_process(df, [1, 3, 7, 14, 30, 60, 90])
    data = data_cleaning(data)
    data = label_process(data)

    data = data[~data['y_7'].isna()]  # 排除y的nan值

    X, y = data[['month', 'quarter', 'de_sum_1', 'de_mean_1', 'de_max_1', 'de_min_1', 'de_sum_3', 'de_mean_3', 'de_max_3', 'de_min_3',
                 'de_sum_7', 'de_mean_7', 'de_max_7', 'de_min_7', 'de_sum_14', 'de_mean_14', 'de_max_14', 'de_min_14',
                 'de_sum_30', 'de_mean_30', 'de_max_30', 'de_min_30', 'de_sum_60', 'de_mean_60', 'de_max_60', 'de_min_60',
                 'de_sum_90', 'de_mean_90', 'de_max_90', 'de_min_90',]], data['y_7']

    xgb_reg = xgboost.XGBRegressor()
    xgb_reg.fit(X, y)
    y_pred = xgb_reg.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f'mes: {mse}, r2: {r2}')

    scatter_plot_with_diagonal(y, y_pred)

