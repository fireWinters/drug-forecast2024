'''
Author: callus
Date: 2024-02-03 16:59:27
LastEditors: callus
Description: some description
FilePath: /drug-forecast2024/AddGrugCode.py
'''
import pandas as pd

# 读取mergedData2020.csv文件
merged_data = pd.read_csv('mergedData2020.csv')

# 读取ClassificationOfWesternMedicineFullNew.csv文件
classification_data = pd.read_csv('ClassificationOfWesternMedicineFullNew.csv')

# 合并两个文件，根据药品名称进行匹配
merged_data_with_classification = pd.merge(merged_data, classification_data, left_on='药品名称', right_on='药品名称', how='left')

# 提取药品分类代码列的值，并填充到新的"药品分类代码"列中
merged_data_with_classification['药品分类代码'] = merged_data_with_classification['药品分类代码_y']

# 删除多余的列
merged_data_with_classification = merged_data_with_classification.drop(['药品分类代码_x', '药品分类代码_y'], axis=1)

# 保存修改后的DataFrame回CSV文件
merged_data_with_classification.to_csv('mergedData2020_with_classification.csv', index=False)
