'''
Author: Diana tang 1987567948@qq.com
Date: 2024-03-03 20:39:06
LastEditors: Diana tang 1987567948@qq.com
LastEditTime: 2024-03-03 21:01:06
FilePath: \drug-forecast2024\findDifferentDrugNames.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import pandas as pd

# 获取XGBoostImg文件夹下所有图片的名字
folder_path = 'XGBoostImg'
file_names = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.png')]

# 提取文件名中"xgboost_"之后、".img"之前的内容并保存在一个CSV文件中

extracted_names = []
for file_name in file_names:
    start_index = file_name.find('xgboost_') + len('xgboost_')
    end_index = file_name.find('.png')
    extracted_name = file_name[start_index:end_index]
    extracted_names.append(extracted_name)

print(extracted_names,'文件名提取完成')
# 读取药品名称文件
drug_names_df = pd.read_csv('./filtered_data2.csv')
drug_names = drug_names_df['药品分类代码'].tolist()
print(drug_names,'药品名称文件读取完成')

# 对比文件中的名称，将extracted_names里没有的药品分类代码保存为新CSV文件

different_names = [name for name in drug_names if name not in extracted_names]
print(different_names,'不同的药品分类代码提取完成')
different_names_df = pd.DataFrame({'药品分类代码': different_names})
different_names_df.to_csv('./不同的药品分类代码.csv', index=False)



# import os

# # 获取XGBoostImg文件夹下所有文件的名字
# folder_path = 'XGBoostImg'
# file_names = os.listdir(folder_path)

# # 提取文件名中"xgboost_"之后、".img"之前的内容并保存在一个文件中
# with open('extracted_names.txt', 'w') as f:
#     for file_name in file_names:
#         start_index = file_name.find('xgboost_') + len('xgboost_')
#         end_index = file_name.find('.img')
#         extracted_name = file_name[start_index:end_index]
#         f.write(extracted_name + '\n')

# # 读取药品名称文件
# with open('药品名称文件.txt', 'r') as f:
#     drug_names = f.read().splitlines()

# # 对比文件中的名称，将不同的药品名称保存为新文件
# different_names = [name for name in drug_names if name not in extracted_names]
# with open('不同的药品名称.txt', 'w') as f:
#     for name in different_names:
#         f.write(name + '\n')

