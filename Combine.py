# -*- coding: utf-8 -*-
import os
import pandas as pd

# 文件夹路径
folder_path = 'D:\\BD_project\\output_combined.csv'
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# 合并所有CSV文件
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

# 保存为单个CSV文件
output_file = '\\combined_output.csv'  
df.to_csv(output_file, index=False)