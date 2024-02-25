import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import xgboost
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
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
# 把xgboost的模型预测展示在折线图，真实值和预测值
def scatter_plot_with_xgboost(y_true, y_pred):
    # 设置agg.path.chunksize的值大于100
    mpl.rcParams['agg.path.chunksize'] = 200
   # 绘制预测值和真实值的折线对比图
    plt.figure(figsize=(8, 8))
    plt.plot(y_true, label='True values', marker='o',color='blue')
    plt.plot(y_pred, label='Predicted values', marker='x',color='red')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('XGBoost Predicted vs True Values')
    plt.legend()
    plt.show()
    # plt.figure(figsize=(8, 6))
    # plt.plot(y_true, label='True values', marker='o')
    # plt.plot(y_pred, label='Predicted values', marker='x')
    # plt.xlabel('Sample')
    # plt.ylabel('Value')
    # plt.title('XGBoost Predicted vs True Values')
    # plt.legend()
    # plt.grid(True)
    # plt.show() 

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
                # 增加一列日期，值为sheet_name和文件名的组合
                year_name=filename.split('.')[0]
                date_str = f'{year_name}-{sheet_name.replace(".", "-")}'  # 将点号替换为横杠，构造日期字符串
                sheet_data['日期'] = datetime.strptime(date_str, '%Y-%m-%d').date()
                # 添加到all_dataframes列表中
                all_dataframes.append(sheet_data)

    # 将所有数据拼接成一个DataFrame
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)

    final_dataframe = final_dataframe.drop(['入库单位', '基本单位'], axis=1)
    # 输出处理后的数据为CSV格式文件
    # final_dataframe.to_csv('final_dataframe.csv', index=False)

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

# Todo 药品类别,在数据中增加一列：药品分类代码，药品名称两文件相同的，药品分类代码用同一个代码
def drug_category(data,fileB):
    # 读取A文件和B文件
    df_A = fileB
    # 关键药品名称列
    df_B=data
    df_B['药品名称'] = data['药品名称'].str.replace(r'\(.*\)', '')  # 去掉括号及括号内的内容
    # 将文件中药品名称列和药品分类代码列提取出来，构建字典
    mapping_dict = dict(zip(df_A['药品名称'].astype(str), df_A['药品分类代码']))
    # 定义一个函数，根据药品名称在字典中查找对应的药品分类代码，如果药品名称中包含字典中的药品名称，则返回对应的药品分类代码，否则返回空字符串
    def get_category(name):

        for key in mapping_dict.keys():
            if key in name:
                return mapping_dict[key]
        return ''

    # 在Data中新增一列，根据药品名称列查找对应的药品分类代码
    data['药品分类代码'] = data['药品名称'].apply(get_category)

    # 将处理后的B文件保存为新csv文件
    data.to_csv('B_with_category_new.csv', index=False)
    return data



# 首先指定了数据文件夹路径，然后调用get_all_dataframes函数获取所有数据，并进行数据处理和特征工程。接着使用XGBoost模型训练数据，并计算模型的均方误差（MSE）和决定系数（R^2）。最后绘制了真实值和预测值的散点图。
if __name__ == '__main__':
    current_directory = r'/Users/callustang/tangCode/shantouCode/drug-forecast2024'
    data_folder_path = os.path.join(current_directory, 'realData')

    df = get_all_dataframes()

    data = data_process(df, [1, 3, 7, 14, 30, 60, 90])
    data = data_cleaning(data)
    data = label_process(data)
    # 读取药品分类代码文件
    fileB = pd.read_csv('./ClassificationOfWesternMedicineFullNew.csv')
    data = drug_category(data,fileB)

    data = data[~data['y_7'].isna()]  # 排除y的nan值
    # y_7数据大于10000的，剔除
    data = data[data['y_7'] < 10000]  
    X, y = data[['month', 'quarter', 'de_sum_1', 'de_mean_1', 'de_max_1', 'de_min_1', 'de_sum_3', 'de_mean_3', 'de_max_3', 'de_min_3',
                 'de_sum_7', 'de_mean_7', 'de_max_7', 'de_min_7', 'de_sum_14', 'de_mean_14', 'de_max_14', 'de_min_14',
                 'de_sum_30', 'de_mean_30', 'de_max_30', 'de_min_30', 'de_sum_60', 'de_mean_60', 'de_max_60', 'de_min_60',
                 'de_sum_90', 'de_mean_90', 'de_max_90', 'de_min_90',]], data['y_7']
    # 导出处理后的数据为CSV文件
    # data.to_csv('label_data.csv', index=False)
    # 导出带有药品分类代码的数据为CSV文件
    # data.to_csv('label_data_with_category.csv', index=False)
      
    # data.to_csv('label_data_lessthan10000.csv', index=False)
    # xgb_reg = xgboost.XGBRegressor()
    # xgb_reg.fit(X, y)
    # y_pred = xgb_reg.predict(X)

    # mse = mean_squared_error(y, y_pred)
    # r2 = r2_score(y, y_pred)
    # print(f'mes: {mse}, r2: {r2}')

    # scatter_plot_with_diagonal(y, y_pred)
    # mes: 1098696.5627478997, r2: 0.80858310294179

  
    # 训练集和测试集的划分，使用train_test_split函数，将数据集划分为训练集和测试集，其中测试集占30%。   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 初始化XGBRegressor
    # model = xgboost.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                    # max_depth = 5, alpha = 10, n_estimators = 10)
    # model=xgboost.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.9, learning_rate = 0.1,
    #                 max_depth = 5, alpha = 10, n_estimators = 10)
    model=xgboost.XGBRegressor()
    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    predictions = model.predict(X_test)

    # 模型评估，使用均方根误差（RMSE）、MES和R^2来评估模型的性能。
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse=mean_squared_error(y_test, predictions)
    r2=r2_score(y_test, predictions)
    print("RMSE: %f" % (rmse), "MSE: %f" % (mse), "R^2: %f" % (r2))
    # 没有剔除大于10000的数据时的结果：RMSE: 1678.441078 MES: 2817164.452318 R^2: 0.442721
    # RMSE: 1678.441078 MES: 2817164.452318 R^2: 0.442721
    # 剔除>10000数据
    # RMSE: 498.584804 MSE: 248586.806396 R^2: 0.540677
    scatter_plot_with_diagonal(y_test, predictions)
    #colsample_bytree = 0.7 RMSE: 490.095843 MSE: 240193.934989 R^2: 0.556185
    #colsample_bytree = 0.9 RMSE: 489.531457 MSE: 239641.047723 R^2: 0.557206
    # scatter_plot_with_xgboost(y_test, predictions)


