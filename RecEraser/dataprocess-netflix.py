from DRAGRU.recbole_utils import RecUtils

MODEL = "BPR"
# 处理的数据集
DATASET = "netflix-process"

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

import pandas as pd

# 全局缓存，用于存储 user_id 到编号的映射
user_id_cache = {}


def remap_user_id(df: pd.DataFrame):
    """
    重新为 DataFrame 中的 'user_id' 列编号，按照从小到大的顺序，并且
    使用全局缓存来维护 user_id 到编号的映射关系。

    参数:
    df: 输入的 pandas DataFrame，必须包含 'user_id' 列。

    返回:
    df: 重新编号后的 DataFrame。
    """

    # 检查是否已经存在缓存的映射
    global user_id_cache

    # 获取当前 DataFrame 中的所有 user_id
    unique_user_ids = df['user_id'].unique()

    # 初始化一个列表用于存储新的 user_id 编号
    new_user_ids = []

    # 处理每个唯一的 user_id
    for user_id in unique_user_ids:
        if user_id not in user_id_cache:
            # 如果缓存中没有该 user_id，则递增编号并保存到缓存
            user_id_cache[user_id] = len(user_id_cache)

        # 将映射关系中的编号添加到新的列表中
        new_user_ids.append(user_id_cache[user_id])

    # 创建新的 'user_id' 列并替换原有列
    df['user_id'] = df['user_id'].map(lambda x: user_id_cache[x])

    return df

inter_df_train = remap_user_id(inter_df_train)
inter_df_test = remap_user_id(inter_df_test)
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