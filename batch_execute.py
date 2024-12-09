import subprocess
import os
import gc
import torch
import numpy as np
import re
import time
DATASET = "ml-100k"
MODEL = "BPR"
config_files = "config_file/ml-100k.yaml"

models = []
K = 5
ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for filename in os.listdir("saved/"):
    models.append(os.path.join("saved/", filename))
models = sorted(models)

count = 0
result = np.zeros((K, len(ratios)), dtype=float)

for index, ratio in enumerate(ratios):
    for label in range(K):
        start_time = time.time()


        dataset_name = f"{DATASET}-{int(ratio * 100)}-{label}"

        if count < len(models):
            # 构造命令行
            command = [
                "/enter/envs/rella/bin/python", "/root/haichao/LLM_rec_unlearning/DRAGRU/movie-lens/data_process/train.py",  # 脚本路径
                "--model", MODEL,  # 输入文件
                "--dataset", dataset_name,  # 输出文件
                "--model_file", models[count], # 输出文件
                "--config_files",config_files
            ]
        else:
            # 构造命令行
            command = [
                "/enter/envs/rella/bin/python",
                "/root/haichao/LLM_rec_unlearning/DRAGRU/movie-lens/data_process/train.py",  # 脚本路径
                "--model", MODEL,  # 输入文件
                "--dataset", dataset_name,  # 输出文件
                "--config_files", config_files
            ]
        # 执行命令
        return_result = subprocess.run(command, capture_output=True, text=True)

        count = count+1

        result[label, index] = float(re.sub(r'\x1b\[[0-9;]*[mK]', '', return_result.stdout.strip()))
        end_time = time.time()

        print(f"finished: {dataset_name}, result: {return_result.stdout}, 代码执行时间: {end_time - start_time:.6f} 秒")
        torch.cuda.empty_cache()
        gc.collect()

print(result)

from texttable import Texttable
import csv
# 创建表格对象
table = Texttable()
table.set_cols_align(["l"] + ["c"] * 10)  # 第一列左对齐，其余列居中对齐
table.set_cols_valign(["m"] * 11)  # 所有列都垂直居中

# 构建表头
header = [""] + [f"{i * 5}%" for i in range(1, 11)]  # 从5%到50%
table.add_row(header)

# 添加矩阵内容到表格中
for i in range(K):
    row = [f"category{i + 1}"] + list(result[i])  # 每一行的分类加上对应的矩阵值
    table.add_row(row)

# 打印表格
print(table.draw())

# 将矩阵写入 CSV 文件
with open(f'{DATASET}-{MODEL}-{K}-result.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(result)  # 直接写入矩阵的每一行
