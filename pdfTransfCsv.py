'''
Author: callus
Date: 2024-02-03 15:54:11
LastEditors: callus
Description: some description
FilePath: /drug-forecast2024/pdfTransfCsv.py
'''
import tabula

# 从PDF文件中提取表格数据
df = tabula.read_pdf('ClassificationOfWesternMedicine.pdf', pages='all')

# 将DataFrame保存为CSV文件
df.to_csv('ClassificationOfWesternMedicine.csv', index=False)
