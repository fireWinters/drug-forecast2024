'''
Author: callus
Date: 2024-01-27 23:57:56
LastEditors: callus
Description: 这个文件是将所有的原始数据合并到一个文件中，把xcel文件中的每个sheet页数据合并到一个DataFrame中，然后将所有sheet页的数据合并到一个DataFrame中，最后将合并后的数据导出为CSV文件。
FilePath: /drug-forecast2024/mergeSheetPage.py
'''
import os
import pandas as pd

# 定义文件夹路径和输出文件名
# folder_path = 'realData/'
folder_path = 'testData/'
# output_file = 'mergedData.csv'
output_file = 'mergedData2020.csv'

# 创建一个空的DataFrame用于存储合并后的数据
merged_data = pd.DataFrame()

# 遍历文件夹中的所有Excel文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        
        # 读取Excel文件中的所有sheet页数据
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        # 遍历每个sheet页，将数据合并到merged_data中
        for sheet_name, sheet_data in excel_data.items():
            # 添加日期列
            sheet_data['日期'] = file_name.split('.')[0] + '.' + sheet_name
            
            # 将当前sheet页数据添加到merged_data中
            merged_data = merged_data._append(sheet_data, ignore_index=True)

# 导出合并后的数据为CSV文件
merged_data.to_csv(output_file, index=False)
