import pandas as pd
import numpy as np
import os

# 假设 DATASET 是一个目录，包含用户、物品和交互数据
# 创建保存目录
output_dir = "dataset"
DATASET = "ml-100k"


items = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.item"), delimiter='\t')
user_data = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.user"), delimiter='\t')
inter_data = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.inter"), delimiter='\t')

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 切割比例
ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# 生成切割后的数据集
for ratio in ratios:
    # 随机抽样 inter_data 中的交互数据
    inter_sampled = inter_data.sample(frac=ratio, random_state=42)

    # 生成保存路径
    output = os.path.join(output_dir, f'{DATASET}-{int(ratio * 100)}')
    os.makedirs(output, exist_ok=True)

    # 保存切割后的交互数据
    inter_sampled.to_csv(os.path.join(output, f'{DATASET}-{int(ratio * 100)}.inter'), sep='\t', index=False)
    user_data.to_csv(os.path.join(output, f'{DATASET}-{int(ratio * 100)}.user'), sep='\t', index=False)
    items.to_csv(os.path.join(output, f'{DATASET}-{int(ratio * 100)}.item'), sep='\t', index=False)

    print(f"Generated: {ratio}")

# 输出成功消息
print("切割后的数据集已生成！")
