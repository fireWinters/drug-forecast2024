'''
Author: callus
Date: 2024-02-23 14:43:07
LastEditors: callus
Description: 这个函数用于获取指定文件夹中的所有xlsx文件，并将每个文件中的每个sheet拼接成一个DataFrame。
FilePath: /drug-forecast2024/get_all_dataframes.py
'''
# 这个函数用于获取指定文件夹中的所有xlsx文件，并将每个文件中的每个sheet拼接成一个DataFrame。
def get_all_dataframes():
    """获取原始数据下excel，并将每一个sheet拼接成一个df"""
    all_dataframes = []
    # 遍历文件夹中的每个xlsx文件
    for filename in tqdm(os.listdir(data_folder_path)):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(data_folder_path, filename)

            # 读取xlsx文件的所有sheet
            sheets_dict = pd.read_excel(file_path, sheet_name=None)

            # 遍历每个sheet
            for sheet_name, sheet_data in sheets_dict.items():
                # 增加一列日期，值为sheet_name和文件名的组合
                year_name=filename.split('.')[0]
                date_str = f'{year_name}-{sheet_name.replace(".", "-")}'  # 将点号替换为横杠，构造日期字符串
                sheet_data['日期'] = datetime.strptime(date_str, '%Y-%m-%d').date()
                # 添加到all_dataframes列表中
                all_dataframes.append(sheet_data)

    # 将所有数据拼接成一个DataFrame
    final_dataframe = pd.concat(all_dataframes, ignore_index=True)

    final_dataframe = final_dataframe.drop(['入库单位', '基本单位'], axis=1)
    # 输出处理后的数据为CSV格式文件
    final_dataframe.to_csv('final_dataframe.csv', index=False)

    return final_dataframe
