import pandas as pd
# 读取文件
df = pd.read_csv('./all_category_new.csv')

# 选择要筛选的列名
column_name = '药品分类代码'
# 删除药品分类代码列数据相同的行
df = df.drop_duplicates(subset=column_name)
df.to_csv('./filtered_data2.csv', index=False)
print('筛选完成')