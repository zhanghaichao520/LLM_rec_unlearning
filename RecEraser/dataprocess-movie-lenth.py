from DRAGRU.recbole_utils import RecUtils

MODEL = "BPR"
# 处理的数据集
DATASET = "ml-100k"

# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
config_files = f"config_file/{DATASET}.yaml"
config = {"normalize_all": False}
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)

DATASET_REMAIN = f"{DATASET}-remain"
# 目标文件路径
train_file_path = f'data/{DATASET_REMAIN}/train.txt'
test_file_path = f'data/{DATASET_REMAIN}/test.txt'

rec_utils_remain = RecUtils(model=MODEL, dataset=DATASET_REMAIN, config_file_list=config_file_list, config_dict=config)
inter_df_train = rec_utils_remain.ori_trainset
inter_df_test = rec_utils_remain.ori_testset

def convert_format_and_dump(df, filename):

    # 按user_id:token分组，聚合所有item_id:token
    grouped = df.groupby('user_id')['item_id'].apply(list)
    # 按user_id:token排序
    grouped_sorted = grouped.sort_index()
    # 写入txt文件
    with open(filename, 'w') as f:
        for user, items in grouped_sorted.items():
            # 将用户ID和所有item_id连接成一行，用户和item之间用空格隔开
            line = f"{user} " + " ".join(map(str, items)) + "\n"
            f.write(line)

    print("文件写入完成")

import os


# 确保目标目录存在，如果不存在则创建
os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

convert_format_and_dump(inter_df_train, train_file_path)
convert_format_and_dump(inter_df_test, test_file_path)