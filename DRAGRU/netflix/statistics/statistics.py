## 统计训练集里面的交互平均次数， 用来指导大模型的历史交互数量参数设置

import pandas as pd
from DRAGRU.recbole_utils import RecUtils

MODEL = "LightGCN"
# 处理的数据集
DATASET = "yelp-2018"
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
config = {"normalize_all": False}
# config_files = None
config_files = "config_file/yelp-2018.yaml"
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)

rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict = config)

df = rec_utils.ori_trainset

# 统计每个用户的交互次数
interaction_counts = df.groupby("user_id").size()

# 计算平均值和分位数
mean_interactions = interaction_counts.mean()
quantiles = interaction_counts.quantile([0.2, 0.5, 0.8, 0.9])

# 输出结果
print(f"平均交互次数: {mean_interactions:.2f}")
print("分位数统计:")
print(quantiles)
