import subprocess
import os
import gc
import torch
import numpy as np
import re
import time
DATASET = "netflix-process-remain"
MODEL = "BPR"
config_files = "config_file/netflix-process.yaml"

split_num = 3
for idx in range(0, split_num):
    start_time = time.time()

    dataset_name = f"{DATASET}-SISA-part{idx}"

    # 构造命令行
    command = [
        "/enter/envs/rella/bin/python",
        "/root/haichao/LLM_rec_unlearning/SISA/train_single_model.py",  # 脚本路径
        "--model", MODEL,  # 输入文件
        "--dataset", dataset_name,  # 输出文件
        "--config_files", config_files
    ]
    # 执行命令
    return_result = subprocess.run(command, capture_output=True, text=True)

    end_time = time.time()

    print(f"finished: {dataset_name}, result: {return_result.stdout}, 代码执行时间: {end_time - start_time:.6f} 秒")
    torch.cuda.empty_cache()
    gc.collect()



