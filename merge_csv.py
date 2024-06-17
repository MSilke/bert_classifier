import os
import pandas as pd
import numpy as np

# 指定文件夹路径
folder_path = 'mr_dataset'  # 替换为你的CSV文件所在的文件夹路径
output_filename = 'dataset.csv'  # 合并后的CSV文件名

# 创建一个空的列表，用于存储所有DataFrame
all_files = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # 检查文件是否为CSV文件
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 读取CSV文件到DataFrame
        df = pd.read_csv(file_path, header=None)
        # 将DataFrame添加到列表中
        all_files.append(df)

    # 使用concat方法合并所有的DataFrame
all_data = pd.concat(all_files, axis=0, ignore_index=True)

# 将合并后的数据保存到新的CSV文件中
all_data.to_csv(os.path.join(folder_path, output_filename), index=False, header=False)

print(f"All CSV files in {folder_path} have been merged into {output_filename}")
