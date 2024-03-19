# 从A .csv文件中，筛选出药品分类代码和另一个csv文件相同的全部数据并导出为C.csv文件

import pandas as pd

# 读取两个CSV文件的数据
df1 = pd.read_csv('./category_data_all_S_noNan0228.csv')
df2 = pd.read_csv('./不同的药品分类代码20240305.csv')

# 根据药品分类代码将两个数据集进行匹配
merged_df = pd.merge(df1, df2, on='药品分类代码', how='inner')

# 筛选出匹配的数据
# result_df = merged_df[['药品分类代码', '其他字段1', '其他字段2']]  # 根据实际情况选择需要的字段

# 将筛选后的数据导出为一个新的CSV文件
merged_df.to_csv('C.csv', index=False)
