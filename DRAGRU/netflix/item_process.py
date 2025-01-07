import os
import pandas as pd

# 创建保存目录
output_dir = "dataset"
DATASET = "netflix"
DATASET_PROCESS = "netflix-process"
# 加载数据
inter_df = pd.read_csv(f'dataset/{DATASET}/{DATASET}.inter', delimiter='\t')
# user_df = pd.read_csv(f'dataset/{DATASET}/{DATASET}.user', delimiter='\t')
item_df = pd.read_csv(f'dataset/{DATASET}/{DATASET}.item', delimiter='\t')

print(len(inter_df))
import pandas as pd

# 假设你已经加载了inter_df
# Step 1: 统计每个用户的交互记录数
user_interaction_count = inter_df.groupby('user_id:token').size()

# Step 2: 筛选交互记录数在200到500之间的用户
valid_users = user_interaction_count[(user_interaction_count >= 200) & (user_interaction_count <= 210)].index

# Step 3: 根据筛选出的用户ID，提取对应的记录
filtered_inter_df = inter_df[inter_df['user_id:token'].isin(valid_users)]
# filtered_user_df = user_df[user_df['user_id:token'].isin(valid_users)]
# filtered_item_df = item_df[item_df['user_id:token'].isin(valid_users)]

# Step 4: 保存数据到文件
def save_dataframe(df, filename):
    df.to_csv(os.path.join(output_dir, filename), sep='\t', index=False)

os.makedirs(f"{output_dir}/{DATASET_PROCESS}", exist_ok=True)

# save_dataframe(filtered_user_df, f'{DATASET_PROCESS}/{DATASET_PROCESS}.user')
save_dataframe(item_df, f'{DATASET_PROCESS}/{DATASET_PROCESS}.item')
save_dataframe(filtered_inter_df, f'{DATASET_PROCESS}/{DATASET_PROCESS}.inter')

print(f"item_df len: {len(item_df)}")
print(f"filtered_inter_df len: {len(filtered_inter_df)}")
print(f"filtered_inter_df user len: {len(filtered_inter_df['user_id:token'].unique())}")
print(f"filtered_inter_df item len: {len(filtered_inter_df['item_id:token'].unique())}")