import pandas as pd
import numpy as np
import os

# 假设 DATASET 是一个目录，包含用户、物品和交互数据
# 创建保存目录
output_dir = "dataset"
# 直接切remain集合， 和切原始集合在删掉forget集的效果一样
ORI_DATASET = "amazon-all-beauty-18"
DATASET = f"{ORI_DATASET}-remain"

# 切割比例
split_num = 10
items = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.item"), delimiter='\t')
# user_data = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.user"), delimiter='\t')
inter_data = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.inter"), delimiter='\t')
forget_inter_data = pd.read_csv(os.path.join(f"{output_dir}/{ORI_DATASET}-forget", f"{ORI_DATASET}-forget.inter"), delimiter='\t')

seed = 42
# 设置随机种子以确保结果可重复
np.random.seed(seed)

# inter_data = inter_data.sample(frac=1, random_state=seed).reset_index(drop=True)
# 获取去重后的 user_id 和 item_id
unique_user_ids = inter_data['user_id:token'].unique()
unique_item_ids = inter_data['item_id:token'].unique()

split_size = len(inter_data) // split_num

# 将数据切割成10等份
splits = [inter_data.iloc[i*split_size:(i+1)*split_size] for i in range(split_num)]

# 如果数据不能完全整除10等份，处理剩余数据
remaining_data = inter_data.iloc[split_num*split_size:]
if len(remaining_data) > 0:
    splits[-1] = pd.concat([splits[-1], remaining_data])

import os
need_retrain_split = []
# 生成切割后的数据集
for idx, split in enumerate(splits):
    print(f"Processing split {idx}")
    # 2.1 获取当前分割后的数据的去重 user_id 和 item_id
    current_user_ids = split['user_id:token'].unique()
    current_item_ids = split['item_id:token'].unique()
    print(f"Before Unique user_id count: {len(current_user_ids)}")
    print(f"Before Unique item_id count: {len(current_item_ids)}")

    # 2.2 计算缺失的 user_id 和 item_id
    missing_user_ids = np.setdiff1d(unique_user_ids, current_user_ids)
    missing_item_ids = np.setdiff1d(unique_item_ids, current_item_ids)

    # 2.3 补齐缺失的 user_id 和 item_id
    new_rows = []

    # 为缺失的user_id和item_id从inter_data中补齐数据
    for user_id in missing_user_ids:
        # 从inter_data中找到对应的user_id的任意一条数据进行补充
        row = inter_data[inter_data['user_id:token'] == user_id].iloc[0]
        new_rows.append(row)

    for item_id in missing_item_ids:
        # 从inter_data中找到对应的item_id的任意一条数据进行补充
        row = inter_data[inter_data['item_id:token'] == item_id].iloc[0]
        new_rows.append(row)

    # 如果有新行，则使用 pd.concat 来补齐
    if new_rows:
        new_data = pd.DataFrame(new_rows)
        split = pd.concat([split, new_data], ignore_index=True)
    # 2.4 打印去重后的 item_id 和 user_id 个数
    print(f"Unique user_id count: {len(split['user_id:token'].unique())}")
    print(f"Unique item_id count: {len(split['item_id:token'].unique())}")
    print(f"Unique inter count: {len(split)}")

    # 使用 merge 查找交集
    intersection = pd.merge(forget_inter_data, split, how='inner')
    if len(intersection) > 1000:
        need_retrain_split.append(idx)
    dataset_name = f"{DATASET}-SISA-part{idx}"
    # 生成保存路径
    output = os.path.join(output_dir, f'{dataset_name}')
    os.makedirs(output, exist_ok=True)
    # 保存切割后的交互数据
    split.to_csv(os.path.join(output, f'{dataset_name}.inter'), sep='\t', index=False)
    # user_data.to_csv(os.path.join(output, f'{dataset_name}.user'), sep='\t', index=False)
    items.to_csv(os.path.join(output, f'{dataset_name}.item'), sep='\t', index=False)

    print(f"Generated dataset: {dataset_name}, inter len: {len(split)}")
    print("\n\n")

# 输出成功消息
print("切割后的数据集已生成！")
print(f"需要重新训练的分片有！{need_retrain_split}")
