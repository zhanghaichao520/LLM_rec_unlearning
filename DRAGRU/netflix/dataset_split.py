## 按照用户切分forget集和remain集
import os
import pandas as pd

# 创建保存目录
output_dir = "dataset"

DATASET = "netflix-process"
ratio = 0.1
DATASET_forget = f"{DATASET}-forget"
DATASET_remain = f"{DATASET}-remain"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/{DATASET_forget}", exist_ok=True)
os.makedirs(f"{output_dir}/{DATASET_remain}", exist_ok=True)


item_attr_path = os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.item")
item_attr_df = pd.read_csv(item_attr_path, delimiter='\t')

inter_path = os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.inter")
inter_df = pd.read_csv(inter_path, delimiter='\t')

print("*" * 30)
print("origin dataset size:")
print(f"item_attr_df : {len(item_attr_df)}")
print(f"inter_df : {len(inter_df)}")
print(f"inter item count : {len(inter_df['item_id:token'].unique())}")
print("*" * 30)

import pandas as pd
from tqdm import tqdm
import random


def split_interactions_with_user_logic(interaction_df, frac=0.2, random_state=42):
    # 设置随机种子
    random.seed(random_state)

    # 获取所有用户
    all_users = interaction_df['user_id:token'].unique()

    # 随机选择一部分用户作为目标用户，将其所有交互放到 Forget Set 中
    num_users_to_forget = int(len(all_users) * frac)
    forget_users = random.sample(list(all_users), num_users_to_forget)

    # 初始化 Remain Set 和 Forget Set
    remain_set = []
    forget_set = []

    # Step 1: 将被选中用户的所有交互转移到 Forget Set，并保留一条交互在 Remain Set
    for user_id in tqdm(forget_users, desc="Processing forget users"):
        user_interactions = interaction_df[interaction_df['user_id:token'] == user_id]

        # 全部交互转移到 Forget Set
        forget_set.extend(user_interactions.to_dict('records'))

        # 保留一条交互在 Remain Set
        remain_set.append(user_interactions.iloc[0].to_dict())

    # Step 2: 处理剩余用户，确保每个用户至少有一条交互进入 Forget Set
    remaining_users = set(all_users) - set(forget_users)
    for user_id in tqdm(remaining_users, desc="Processing remaining users"):
        user_interactions = interaction_df[interaction_df['user_id:token'] == user_id]

        # 添加所有交互到 Remain Set
        remain_set.extend(user_interactions.to_dict('records'))

        # 从中随机选择一条交互加入 Forget Set
        forget_set.append(user_interactions.sample(1, random_state=random_state).iloc[0].to_dict())

    # 转换为 DataFrame
    remain_set = pd.DataFrame(remain_set)
    forget_set = pd.DataFrame(forget_set)

    # Step 3: 统计用户交互数量
    forget_user_stats = forget_set['user_id:token'].value_counts()
    remain_user_stats = remain_set['user_id:token'].value_counts()

    forget_count_1 = (forget_user_stats == 1).sum()
    forget_count_gt1 = (forget_user_stats > 1).sum()

    remain_count_1 = (remain_user_stats == 1).sum()
    remain_count_gt1 = (remain_user_stats > 1).sum()

    stats = {
        "forget_count_1": forget_count_1,
        "forget_count_gt1": forget_count_gt1,
        "remain_count_1": remain_count_1,
        "remain_count_gt1": remain_count_gt1
    }

    return remain_set, forget_set, stats

# 调用函数
remain_set, forget_set, stats = split_interactions_with_user_logic(inter_df, frac=ratio)

import numpy as np
difference = np.setdiff1d(inter_df["item_id:token"].unique(), remain_set["item_id:token"].unique())

rows = forget_set[forget_set["item_id:token"].isin(difference) ]
remain_set = pd.concat([remain_set, rows], ignore_index=True)

forget_set = forget_set[~forget_set.isin(rows)].dropna(how='all').reset_index(drop=True)

# Step 4: 保存数据到文件
def save_dataframe(df, filename):
    df.to_csv(os.path.join(output_dir, filename), sep='\t', index=False)

save_dataframe(forget_set, f'{DATASET_forget}/{DATASET_forget}.inter')
save_dataframe(item_attr_df, f'{DATASET_forget}/{DATASET_forget}.item')

print("*" * 30)
print("forget dataset size:")
print(f"forget inter count : {len(forget_set)}")
print(f"forget interaction item count : {len(forget_set['item_id:token'].unique())}")

print("*" * 30)

save_dataframe(remain_set, f'{DATASET_remain}/{DATASET_remain}.inter')
save_dataframe(item_attr_df, f'{DATASET_remain}/{DATASET_remain}.item')

print("*" * 30)
print("remain dataset size:")
print(f"remain_interaction : {len(remain_set)}")
print(f"remain_interaction item count : {len(remain_set['item_id:token'].unique())}")
print("*" * 30)

print("数据切分并保存完成！")

print(stats)