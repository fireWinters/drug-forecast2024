'''
Author: Diana tang 1987567948@qq.com
Date: 2024-03-03 15:46:55
LastEditors: Diana tang 1987567948@qq.com
LastEditTime: 2024-03-03 15:47:22
FilePath: \drug-forecast2024\clearnFileNames.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import re

def clean_filename(filename):
    # 使用正则表达式匹配所有非字母数字字符，并替换为下划线
    cleaned_filename = re.sub(r'[^\w\s]', '_', filename)
    
    return cleaned_filename

# 示例使用
original_filename = 'xgboost_羟乙CC基淀粉(130/0.4)氯化钠AABB.png'
cleaned_filename = clean_filename(original_filename)
print(cleaned_filename)


# 给出一段代码，依据药品分类代码，将模型训练的循环从羟乙基淀粉(130/0.4)氯化钠开始，在此之前的药品不做训练
# 依次训练，直到最后一个药品分类代码
# 代码如下：
