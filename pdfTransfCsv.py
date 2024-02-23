'''
Author: callus
Date: 2024-02-03 15:54:11
LastEditors: callus
Description: 处理pdf转换成csv文件，处理药品分类代码列
FilePath: /drug-forecast2024/pdfTransfCsv.py
'''
# from tabula import read_pdf
# import os

# # 指定PDF文件路径
# pdf_path = 'ClassificationOfWesternMedicine.pdf'

# # 指定输出CSV文件路径
# csv_path = 'ClassificationOfWesternMedicine.csv'

# # 检查PDF文件是否存在
# if not os.path.exists(pdf_path):
#     raise Exception(f"The file {pdf_path} does not exist.")

# # 使用tabula的read_pdf函数读取PDF文件中的表格数据
# # 参数pages='all'表示提取所有页面的表格数据
# # 参数multiple_tables=True表示PDF中可能有多个表格
# try:
#     dfs = read_pdf(pdf_path, pages='all', multiple_tables=True, output_format='dataframe')
# except Exception as e:
#     raise Exception(f"An error occurred while reading the PDF file: {e}")

# # 将提取的表格数据（可能有多个DataFrame）保存为CSV文件
# # 如果PDF中有多个表格，它们将被保存到多个CSV文件中
# for i, df in enumerate(dfs):
#     # 如果只有一个表格，直接保存到指定的CSV文件
#     if len(dfs) == 1:
#         df.to_csv(csv_path, index=False)
#     # 如果有多个表格，为每个表格创建一个CSV文件
#     else:
#         csv_file = f"{os.path.splitext(csv_path)[0]}_{i+1}.csv"
#         df.to_csv(csv_file, index=False)
#         print(f"Table {i+1} has been saved to {csv_file}")

# print(f"PDF file {pdf_path} has been successfully converted to CSV format.")


# 把转换后的csv文件中药品分类代码列做处理。遍历药品分类代码一列，如果当前列的单元格为空，则复制上一行有数据的药品代码到当前行，如果不为空则跳过，直到遍历完毕
import pandas as pd
# df = pd.read_csv('ClassificationOfWesternMedicine.csv')
# #检查药品分类代码这一列是否存在
# if '药品分类代码' not in df.columns:
#     raise Exception('The column "药品分类代码" does not exist.')
# ##删除除第一行外的所有包含药品分类代码、药品分类、编号、药品名称、剂型、备注，这些字的行
# # 检查是否存在需要删除的字样行
# keyword = '药品分类代码'
# rows_to_delete = df[df.apply(lambda row: row.str.contains(keyword)).any(axis=1)].index
# df.drop(rows_to_delete)
# df.to_csv('WMAW.csv', index=False)


df = pd.read_csv('WMAW.csv')

df = df[~df['药品分类代码'].str.contains('药品分类代码|药品分类|编号|药品名称|剂型|备注', na=False)]

df = df[df['药品分类代码'] != '药品分类代码']
for i in range(len(df)):
    if pd.isnull(df.loc[i,'药品分类代码']):
        df.loc[i,'药品分类代码'] = df.loc[i-1,'药品分类代码']
df.to_csv('ClassificationOfWesternMedicineFullNew.csv', index=False)
print('处理完成')