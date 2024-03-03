# 导入必要的库
import os  # 用于操作文件系统
import pandas as pd  # 用于数据处理和分析
from datetime import datetime  # 用于处理日期和时间
from tqdm import tqdm  # 用于在循环中显示进度条
import xgboost  # 用于使用XGBoost算法
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error  # 用于模型评估
import matplotlib.pyplot as plt  # 用于绘图
from matplotlib.font_manager import FontProperties
import xgboost as xgb  # 用于使用XGBoost算法
from sklearn.model_selection import cross_val_score, train_test_split  # 用于交叉验证和数据集划分
from bayes_opt import BayesianOptimization  # 用于贝叶斯优化
import lightgbm as lgb  # 用于使用LightGBM算法
from statsmodels.tsa.arima_model import ARIMA  # 用于使用ARIMA模型
import itertools  # 用于创建迭代器
import re

# 定义一个函数，用于绘制真实值和预测值的散点图，并绘制对角线
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

# 定义一个函数，用于获取所有的数据，并将每个sheet拼接成一个DataFrame
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

# # 定义一个函数，用于数据清洗
def data_cleaning(df):
    """把前90/180天没有减少数量的剔除掉"""

    df = df[df['de_sum_90'] > 0]
    # df = df[df['de_sum_180'] > 0]
     # 把药品分类代码列值为删的数据全部剔除
    df = df[df['药品分类代码'] != '删']
    # 删除药品分类代码为空的数据
    df = df[~df['药品分类代码'].isna()]
    return df

# 定义一个函数，用于处理标签
def label_process(df, days=7):
    # 对减少数量进行滚动求和，并且向后移动7天
    df['y'] = df.groupby('药品名称')['减少数量'].transform(
        lambda x: x.rolling(window=days, min_periods=1).sum().shift(-7))

    return df

# 定义一个函数，用于数据处理，包括日期、季度和时序特征
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


# 定义一个函数，用于使用XGBoost算法进行交叉验证
def xgboost_cv_func(data, target, pbounds):
    def xgboost_crossval(max_depth, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
        params = {'max_depth': int(max_depth),
                  'n_estimators': int(n_estimators),
                  'gamma': gamma,
                  'min_child_weight': min_child_weight,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree,
                  'tree_method': 'gpu_hist'
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

# 定义一个函数，用于使用LightGBM算法进行交叉验证
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

# 定义一个函数，用于训练模型
def model_train(X_train, X_test, y_train, y_test, pbounds, model_type):
    if model_type == 'xgboost':
        best_params = xgboost_cv_func(X_train, y_train, pbounds)

        model = xgboost.XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
    elif model_type == 'lightgbm':
        best_params = lightgbm_cv_func(X_train, y_train, pbounds)

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
# 检查是否含有中文
def contains_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False
# 设置中文字体的函数
def set_chinese_font():
    return FontProperties(fname='./fronts/STSONG.TTF')  # 替换为您的中文字体文件路径
# 检查文件名，如果有特殊字符，将特殊字符统一替换为-
def check_filename(filename):
    filename = re.sub(r'[^\w\s]', '_', filename)
    return filename

# 定义一个函数，用于绘制真实值和预测值的折线图，药品分类代码也在图中显示
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
    # 这个是文件名称
    one_title=f'moduleType:{model_type},code:{drug_code},r2: {r2:.2f}, mae: {mae:.2f}, mape: {mape:.2f}, mse: {mse:.2f},name:{drug_name}'
    if contains_chinese(one_title):
        plt.title(one_title, fontproperties=set_chinese_font())
    else:
        plt.title(one_title)
    plt.legend()
    plt.xticks(data['date'], rotation=45)
    plt.tight_layout()
    # 保存图片，用模型名称和药品分类代码命名，如果没有模型名称，用drug名，否则用模型名
    drug_code= check_filename(drug_code)
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


# 主函数
if __name__ == '__main__':
    # current_directory = r'/Users/callustang/tangCode/shantouCode/drug-forecast2024'
    # data_folder_path = os.path.join(current_directory, 'realData')
    # 读取realData文件夹下的所有数据，并将每个sheet拼接成一个DataFrame,用相对路径的方法读取数据
    data_folder_path = r'./realData'
    current_directory = os.path.dirname(__file__)

    # 得到所有数据
    df = get_all_dataframes()
    # 将分类数据与原始数据框架合并，基于'药品名称'列进行左连接
    # 这样可以将'药品分类代码'添加到原始数据框架中
    cate_data = pd.read_csv(os.path.join(current_directory, 'drug_with_category_2024.csv'))
    df = pd.merge(df, cate_data[['药品名称', '药品分类代码']], how='left', on='药品名称')
    # 对数据进行处理，包括日期、季度和时序特征的处理
    # day_lst参数指定了要生成的时序特征的时间窗口长度
    data = data_process(df, [1, 3, 7, 14, 30, 60, 90,180])
    # 对处理后的数据进行清洗，可能包括去除缺失值、异常值或转换数据类型等操作
    data = data_cleaning(data)
    # 数据输出为csv格式
    data.to_csv('./category_data_all_S_noNan.csv', index=False)
    # print(data,'清理后的数据')
    # 对数据进行标签处理，这里的标签可能指的是预测目标
    # 使用未来7天的数据作为标签
    data = label_process(data, 7)
    # 移除标签（预测目标）y中的缺失值，确保模型训练时所有数据都有对应的标签
    data = data[~data['y'].isna()]  # 排除y的nan值
    # 移除标签y中的极端值，这里选择移除99分位数以上的值，这样可以减少极端异常值对模型训练的影响
    data = data[data['y'] < data['y'].quantile(0.99)]  # 排除99分位数以上的值
    # 对数据按照'药品分类代码'进行分组，并计算每个分类的药品数量
    # 这一步可能是为了查看不同分类的药品数量分布
    df.groupby('药品分类代码')['药品分类代码'].count()
    cate_lst = data['药品分类代码'].unique()
    # 将列表cate_lst分成2个列表，分开2列的值是药品分类代码在羟乙基淀粉(130/0.4)氯化钠之前的为一组，之后的为一组
    # 找到"羟乙基淀粉(130/0.4)氯化钠"在列表中的索引
    idx = list(cate_lst).index('羟乙基淀粉(130/0.4)氯化钠')

    # 将列表cate_lst分成两个列表
    cate_lst_before = cate_lst[:idx]
    cate_lst_after = cate_lst[idx+1:]

    print("羟乙基淀粉(130/0.4)氯化钠之前的列表：", cate_lst_before)
    print("羟乙基淀粉(130/0.4)氯化钠之后的列表：", cate_lst_after)
    data = data[data['药品分类代码'] == '羟乙基淀粉(130/0.4)氯化钠']
# 选择特征列和目标列
# 特征包括月份、季度和不同时间窗口（1天、3天、7天、14天、30天、60天、90天）的销售数据统计（总和、平均值、最大值、最小值）
    X, y = data[['month', 'quarter', 'de_sum_1', 'de_mean_1', 'de_max_1', 'de_min_1', 'de_sum_3', 'de_mean_3', 'de_max_3',
         'de_min_3',
         'de_sum_7', 'de_mean_7', 'de_max_7', 'de_min_7', 'de_sum_14', 'de_mean_14', 'de_max_14', 'de_min_14',
         'de_sum_30', 'de_mean_30', 'de_max_30', 'de_min_30', 'de_sum_60', 'de_mean_60', 'de_max_60', 'de_min_60',
         'de_sum_90', 'de_mean_90', 'de_max_90', 'de_min_90', ]], data['y']
# 定义时间外样本（OOT）的索引，这些样本用于模型的最终评估
# 这里选择了日期在2023-6-01之后且药品分类代码为'羟乙基淀粉(130/0.4)氯化钠'的数据作为OOT样本
    # out of time sample
    # 获取系统日期
    pre_today = datetime.now().date()
    # print(pre_today,'今天几号')
    pre_today='2023-05-01'
    # print(pre_today,'给个常量')
    # oot_index = data[(data['日期'] >= '2023-12-01')&(data['药品分类代码'] == '羟乙基淀粉(130/0.4)氯化钠')].index
    # 预测日期 = today + pd.Timedelta(days=7)
    oot_index = data[(data['日期'] >= pre_today) & (data['药品分类代码'] == '羟乙基淀粉(130/0.4)氯化钠')].index
   # 根据OOT索引筛选出对应的特征和目标数据
    oot_x, oot_y = X[X.index.isin(oot_index)], y[oot_index]
    # 得到药品名称和药品分类代码
    print(oot_index,'oot_index')
    drug_name = data.loc[oot_index, '药品名称'].values[0]
    drug_code = data.loc[oot_index, '药品分类代码'].values[0]
# 将数据分割为训练集和测试集，测试集大小为20%，随机种子为100以确保结果可重复
    # 如果OOT没有数据，则代码不再继续执行
    if len(oot_x) == 0:
        print('OOT没有数据',oot_x)
    else:
        # 如果OOT有数据，则继续执行
        # 如果OOT数据比较少，可以考虑增加训练集的大小

        # train & test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
        # return X_train, X_test, y_train, y_test, oot_x, oot_y
# 定义模型超参数的边界，这些超参数将在模型训练过程中进行优化
    pbounds = {
        'max_depth': (3, 8),             # 决策树的最大深度
        'gamma': (0, 1),                 # XGBoost的gamma参数，用于控制是否后剪枝
        'n_estimators': (100, 500),      # 树的数量
        'min_child_weight': (0, 10),     # 决定最小叶子节点样本权重和
        'subsample': (0.5, 1),           # 训练每棵树时样本的抽样比例
        'colsample_bytree': (0.5, 1)     # 训练每棵树时特征的抽样比例
    }
    # 初始化一个列表来存储训练好的模型
    # training models
    model_traineds = []
    # 分别使用XGBoost和LightGBM算法训练模型
    model_types = ['xgboost', 'lightgbm']
    # 对每个模型进行训练
    for model_type in model_types:
        # 训练模型，并返回训练好的模型及其相关信息
        model_trained = model_train(X_train, X_test, y_train, y_test, pbounds, model_type)
        # 将训练好的模型添加到列表中
        model_traineds.append(model_trained)
    # 对每个模型进行评估
    # evaluation models
    for model_trained in model_traineds:
        # 获取模型的相关信息
        model = model_trained['model']
        # 得到模型的类型
        model_type = model_trained['model_type']
        # 使用训练好的模型对OOT样本进行预测
        oot_pred = model.predict(oot_x)
        # 绘制真实值和预测值的折线图
        date = data.loc[oot_index, '日期']
        plot_prediction(date, oot_y, oot_pred, model_type,drug_code,drug_name, save_name=None)

    # p_values = range(0, 6)
    # d_values = range(0, 2)
    # q_values = range(0, 6)
    # pdq_comb = list(itertools.product(p_values, d_values, q_values))
    # evaluate_models(X, pdq_comb)

    # scatter_plot_with_diagonal(y_test, y_pred)
    # scatter_plot_with_diagonal(y_train, y_train_pred)


