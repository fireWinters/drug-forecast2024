'''
Author: callus
Date: 2024-01-28 00:19:13
LastEditors: callus
Description: 格式化CSV数据，删除入库单位和基本单位两列，格式化期初库存和期末库存列为数字列，生成适配XGBoost模型的新的CSV文件。
FilePath: /drug-forecast2024/formatCsvData.py
'''
import pandas as pd

# 读取mergedData.csv文件
data = pd.read_csv('mergedData.csv')

# 删除入库单位和基本单位两列
data = data.drop(['入库单位', '基本单位'], axis=1)

# 格式化期初库存和期末库存列为数字列
data['期初库存'] = data['期初库存'].str.replace(r'[\u4e00-\u9fa5].*', '', regex=True)
data['期末库存'] = data['期末库存'].str.replace(r'[\u4e00-\u9fa5].*', '', regex=True)

# 生成适配XGBoost模型的新的CSV文件
data.to_csv('formattedData.csv', index=False)
