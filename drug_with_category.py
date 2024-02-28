'''
Author: callus
Date: 2024-02-26 14:57:46
LastEditors: callus
Description: 将原始数据中的药品名称和药品分类代码进行匹配，将匹配到的药品分类代码添加到原始数据中
FilePath: /drug-forecast2024/drug_with_category.py
'''
import pandas as pd
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
    data.to_csv('B_with_category_202006.csv', index=False)
    return data

# 读取药品分类代码文件
fileB = pd.read_csv('./ClassificationOfWesternMedicineFullNew.csv')
data=pd.read_excel('/Users/callustang/tangCode/shantouCode/drug-forecast2024/realData/2020.6.xlsx')
data = drug_category(data,fileB)
