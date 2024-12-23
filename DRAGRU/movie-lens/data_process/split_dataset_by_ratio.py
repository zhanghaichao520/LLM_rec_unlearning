import pandas as pd
import numpy as np
import os

# 假设 DATASET 是一个目录，包含用户、物品和交互数据
# 创建保存目录
output_dir = "dataset"
DATASET = "ml-100k"
COL_NAME = "class:token_seq"
items = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.item"), delimiter='\t')
user_data = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.user"), delimiter='\t')
inter_data = pd.read_csv(os.path.join(f"{output_dir}/{DATASET}", f"{DATASET}.inter"), delimiter='\t')

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 切割比例
ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]




import os
import pandas as pd

def get_testdata_by_labels(inter_df, categories):
    filtered_item_ids = []
    for idx, item in items.iterrows():
        for c in categories:
            if c in item[COL_NAME]:
                filtered_item_ids.append(item["item_id:token"])
                break

    return inter_df[inter_df['item_id:token'].isin(filtered_item_ids)]



# 生成切割后的数据集
for ratio in ratios:
    # 随机抽样 inter_data 中的交互数据
    inter_sampled = inter_data.sample(frac=ratio, random_state=42)

    # 遍历 categories_map，打印每个聚类标签和对应的类目列表
    for label, category_list in categories_map.items():
        # print(f"Cluster {label}: {category_list}")
        inter_sampled_labeldata = get_testdata_by_labels(inter_sampled, category_list)

        dataset_name = f"{DATASET}-{int(ratio * 100)}-{label}"
        # 生成保存路径
        output = os.path.join(output_dir, f'{dataset_name}')
        os.makedirs(output, exist_ok=True)
        # 保存切割后的交互数据
        inter_sampled_labeldata.to_csv(os.path.join(output, f'{dataset_name}.inter'), sep='\t', index=False)
        user_data.to_csv(os.path.join(output, f'{dataset_name}.user'), sep='\t', index=False)
        items.to_csv(os.path.join(output, f'{dataset_name}.item'), sep='\t', index=False)

        print(f"Generated dataset: {dataset_name}, inter len: {len(inter_sampled_labeldata)}")

# 输出成功消息
print("切割后的数据集已生成！")
