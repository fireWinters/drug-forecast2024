'''
Author: callus
Date: 2024-01-25 23:12:31
LastEditors: callus
Description: 处理特征值和标签值，将非数字字符转换为数字字符，将处理后的数据导出为Excel文件。
FilePath: /drug-forecast2024/RawDataEigenvalueProcessing.py
'''
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Excel文件路径
file_path = './realData/2020.6.xlsx'

# 读取所有sheet的数据
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names  # 获取所有sheet的名字

# 创建一个空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历每一个sheet
for sheet_name in sheet_names:
    # 读取当前sheet
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # 将当前sheet的数据追加到all_data DataFrame
    all_data = all_data._append(df, ignore_index=True)

# 特征和标签的列名
feature_columns = ['增加数量', '减少数量', '期初库存', '期初金额(进价)', '期初金额(售价)']
label_columns = ['期末库存',
                #   '期末金额(进价)', '期末金额(售价)'
                  ]

# 提取特征和标签
features = all_data[feature_columns]
labels = all_data[label_columns]

# 处理特征中的非数字字符，删除从左到右读，第一个汉字及以后的所有字符，并将结果转换为文本型
qckc = lambda x: re.sub(r'[\u4e00-\u9fa5]+', '', x)
features['期初库存'] = features['期初库存'].apply(lambda x: re.findall(r'\d+', str(qckc))[0] if re.findall(r'\d+', str(qckc)) else 0)
features['增加数量'] = features['增加数量'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else 0)
features['减少数量'] = features['减少数量'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else 0)
features['期初金额(进价)'] = features['期初金额(进价)'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else 0)
features['期初金额(售价)'] = features['期初金额(售价)'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else 0)

#处理标签中的非数字字符
qmck = lambda x: re.sub(r'[\u4e00-\u9fa5]+', '', x)
labels['期末库存'] = labels['期末库存'].apply(lambda x: re.findall(r'\d+', str(qmck))[0] if re.findall(r'\d+', str(qmck)) else 0)
# features['期末金额(进价)'] = features['期末金额(进价)'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else 0)
# features['期末金额(售价)'] = features['期末金额(售价)'].apply(lambda x: re.findall(r'\d+', str(x))[0] if re.findall(r'\d+', str(x)) else 0)


# 特征预处理
#假设所有特征都是数值特征
numeric_features = features.columns.tolist()
# categorical_features = []
# # 假设“库存分类”是分类特征，其他都是数值特征
# numeric_features = features.columns.tolist()
# numeric_features.remove('库存分类')
# categorical_features = ['库存分类']

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 处理数值特征的缺失值
    ('scaler', StandardScaler())])                  # 数值特征标准化

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # 处理分类特征的缺失值
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])                    # 分类特征独热编码

# 使用ColumnTransformer来应用上述变换
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        # ('cat', categorical_transformer, categorical_features)
        ])

#拟合ColumnTransformer
preprocessor.fit(features)

# 应用预处理
features_processed = preprocessor.fit_transform(features)

# 将处理后的特征转换为DataFrame
features_processed_df = pd.DataFrame(features_processed, columns=numeric_features
                                    #   + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
                                      )

# 将特征和标签合并回一个DataFrame
processed_data = pd.concat([features_processed_df, labels.reset_index(drop=True)], axis=1)

# 保存处理后的数据到新文件
output_file_path = './dataC/eigenvalue2_data.xlsx'
processed_data.to_excel(output_file_path, index=False)
