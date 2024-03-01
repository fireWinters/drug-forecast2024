'''
Author: callus
Date: 2024-03-01 21:16:16
LastEditors: callus
Description: some description
FilePath: /drug-forecast2024/deleteDrugCodeFirstletter.py
'''
import pandas as pd
# 文件名称
file_name = 'B_with_category_202402.xlsx'
# 尝试使用不同的编码读取excel文件
# try:
df = pd.read_excel(file_name)
# except:
    # df = pd.read_excel(file_name, encoding='gbk')
# 打印数据的前5行
print(df.head(5))
# 打印数据的形状
print(df.shape)
# 打印所有列名
print(df.columns,'列名')
# 继续处理药品分类代码列
# 如果药品分类代码列的第一个字符是字母X，删除第一个字符
def remove_x_if_starts_with(value):
    # 首先确保值是字符串类型
    value = str(value)
    # 然后检查是否以'X'开头，并相应地处理
    return value[1:] if value.startswith('X') else value

df['药品分类代码'] = df['药品分类代码'].apply(remove_x_if_starts_with)
# 保存处理后的数据到新的CSV文件
df.to_csv('./drug_with_category_2024.csv', index=False)
