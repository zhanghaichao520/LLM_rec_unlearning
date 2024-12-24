from DRAGRU.recbole_utils import RecUtils
import pandas as pd
import os
import json
from tqdm import tqdm
from recbole.utils import set_color
from enum import Enum

class SelectionStrategy(Enum):
    RANDOM = ("Random Selection", 1)
    AVERAGE = ("Average Selection", 2)
    GROUP = ("Group Selection", 3)
    DP = ("DP Selection", 4)

    def __init__(self, description, value):
        self.description = description
        self._value_ = value

def find_model_files(directory_path, model_name):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件名是否包含 "abc"
        if model_name in filename and DATASET in filename:
            return os.path.join(directory_path, filename)

    return None

CANDIDATE_ITEM_NUM = 50
HISTORY_INTER_LIMIT = 100
selection_strategy = SelectionStrategy.DP
# 获取candidate item 的传统推荐模型
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

rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict=config)
rec_utils_remain = RecUtils(model=MODEL, dataset=DATASET_REMAIN, config_file_list=config_file_list, config_dict=config)

MODEL_FILE = find_model_files(directory_path=rec_utils.config["checkpoint_dir"], model_name=MODEL)
# 训练传统模型， 获得模型文件， 用于生成prompt的候选集
if MODEL_FILE is None:
    MODEL_FILE = rec_utils.train()

user_profile_path = os.path.join(rec_utils.config["data_path"], f"{DATASET}.user")
item_attr_path = os.path.join(rec_utils.config["data_path"], f"{DATASET}.item")
inter_path = os.path.join(rec_utils.config["data_path"], f"{DATASET}.inter")

# 加载数据
user_profile_df = pd.read_csv(user_profile_path, delimiter='\t')
item_attr_df = pd.read_csv(item_attr_path, delimiter='\t')
inter_df_remain = rec_utils_remain.ori_trainset

# 创建电影属性字典，方便通过 item_id 查找电影信息
item_attr_dict = {
    str(row["item_id:token"]): {
        "movie_title": row["movie_title:token_seq"],
        "release_year": row["release_year:token"],
        "class": row["class:token_seq"]
    }
    for _, row in item_attr_df.iterrows()
}

#====================item分类================================

# 读取 JSON 格式的文件并将其转换为字典
with open(f'{DATASET}-5-cluster.csv', 'r') as f:
    categories_map = json.load(f)

categories_num = len(categories_map)

#创建分类的逆向映射：便于快速查找每个关键词对应的类别
class_to_category = {}
for category, keywords in categories_map.items():
    for keyword in keywords:
        class_to_category[keyword] = category

def get_item_category(item_info):
    item_classes = item_info['class'].split()  # class:token_seq列是空格分隔的
    categories = []  # 存储匹配到的分类

    # 检查每个类标签是否在字典的分类中
    for item_class in item_classes:
        if item_class in class_to_category:
            categories.extend(class_to_category[item_class])

    return categories


def classify_item(user_history):
    # 将分类结果存储为字典：key 为类别，value 为属于该类别的 item_id 列表
    category_to_items = {str(category): [] for category in categories_map.keys()}
    # 遍历每一行 DataFrame 进行分类
    for _, row in user_history.iterrows():
        item_id = str(row["item_id"])
        categories = get_item_category(item_attr_dict.get(item_id))
        category_to_items[random.choice(categories)].append(item_id)

    return category_to_items


import random

def selection_by_ratio(category_to_items, ratio):
    # 存储最终结果的列表
    selected_items = set()

    # 遍历每个分类及其对应的 item_id 列表
    for category_id, items in category_to_items.items():
        # 获取当前分类的保留比例
        category_ratio = ratio.get(category_id, 0)

        # 根据比例计算需要保留的 item 数量
        num_items_to_select = int(len(items) * category_ratio)

        # 随机选择需要保留的 item_id
        selected_items.update(random.sample(items, num_items_to_select))

    return selected_items


def random_selection(user_history):
    return user_history.sample(n=HISTORY_INTER_LIMIT, random_state=42)


def avg_selection(user_history):
    category_to_items = classify_item(user_history)
    ratio = {str(i): round(HISTORY_INTER_LIMIT/len(user_history), 1) for i in range(1, 6)}
    selected_items = selection_by_ratio(category_to_items, ratio)
    return user_history[user_history['item_id'].isin(selected_items)]

def group_selection(user_history):
    category_to_items = classify_item(user_history)
    ratio = {}
    # 遍历每个分类及其对应的 item_id 列表
    for category_id, items in category_to_items.items():
        if len(items) == 0:
            continue
        category_ratio = round(len(items)/len(user_history), 1)
        ratio[category_id] = min(1.0, round(category_ratio * HISTORY_INTER_LIMIT/len(items), 1))

    selected_items = selection_by_ratio(category_to_items, ratio)
    return user_history[user_history['item_id'].isin(selected_items)]

def dp_selection(user_history):
    category_to_items = classify_item(user_history)
    avg_ratio = round(HISTORY_INTER_LIMIT/len(user_history), 1)
    ratio = {'0': 0.05 * avg_ratio, '1': 0.20 * avg_ratio, '2': 0.15 * avg_ratio,
             '3': 0.25 * avg_ratio, '4': 0.35 * avg_ratio}
    selected_items = selection_by_ratio(category_to_items, ratio)
    return user_history[user_history['item_id'].isin(selected_items)]


#==========================================================

# 生成推荐文本的函数
def generate_recommendation_text(user_id):
    # 获取用户的个人信息
    user_info = user_profile_df[user_profile_df['user_id:token'] == user_id].iloc[0]
    user_profile_text = f"He is {user_info['age:token']} years old, and works as {user_info['occupation:token']}."

    # 从训练集中获取用户的历史观看记录
    user_history = inter_df_remain[inter_df_remain['user_id'] == str(user_id)]
    # 如果用户的交互数量过多， 使用一些策略筛选
    if len(user_history) > HISTORY_INTER_LIMIT:
        # 随机策略
        if selection_strategy == SelectionStrategy.RANDOM:
            user_history = random_selection(user_history)
        # 平均策略
        if selection_strategy == SelectionStrategy.AVERAGE:
            user_history = avg_selection(user_history)
        # 分组策略
        if selection_strategy == SelectionStrategy.GROUP:
            user_history = group_selection(user_history)
        # 动态规划策略（背包优化）
        if selection_strategy == SelectionStrategy.DP:
            user_history = dp_selection(user_history)

    history_movies = []
    for _, row in user_history.iterrows():
        item_info = item_attr_dict.get(str(row["item_id"]))
        if item_info:
            history_movies.append(
                f"a {item_info['class']} movie called {item_info['movie_title']} ({item_info['release_year']}), and scored it {20 * int(row['rating'])}.")

    type = "remain" if len(history_movies) > 2 else "forget"
    history_movies = " \n - ".join(history_movies)

    # 获取推荐电影列表 (仅 item_id 列表)
    candidate_item_ids = rec_utils.get_recommandation_list(ori_user_id=user_id, topk=CANDIDATE_ITEM_NUM, model_file=MODEL_FILE)

    # 根据推荐的 item_id 获取电影名称
    candidate_list = []
    for item_id in candidate_item_ids:
        if item_id in item_attr_dict:
            candidate_list.append(f"{item_id} : {item_attr_dict[item_id]['movie_title']} ({item_attr_dict[item_id]['release_year']}).")
    candidate_list = " \n - ".join(candidate_list)

    prompt = (
        "I want you to predict the user's rating for each movie in the candidate list on a scale from 1 to 100, "
        "based on the user's profile and movie interaction history. Follow these instructions carefully:\n\n"
        "1. Use the given user profile and historical movie interaction records to predict how much the user would "
        "like each movie in the candidate list. The higher the score, the more likely the user will enjoy the movie.\n\n"
        "2. The output must be in valid JSON format, where each movie ID is paired with its predicted score. "
        "The format should be:\n"
        "{\n"
        "  \"movie_id1\": score1,\n"
        "  \"movie_id2\": score2,\n"
        "  ...\n"
        "}\n\n"
        "3. Ensure that all movie IDs in the candidate list are included exactly once in the output.\n\n"
        "4. Do not include any additional text, explanation, or comments outside the JSON object.\n\n"
        "---\n\n"
        "### User Profile:\n"
        f"{user_profile_text}.\n\n"
        "### Movie Interaction History:\n"
        "The user's historical movie interaction records include:\n"
        f"{history_movies}\n"
        "### Candidate List:\n"
        f"{candidate_list}\n"
        "Predict and output the ratings in the required JSON format."
    )

    recommendation_text = {
        "user_id": int(user_id),
        "item_id_list": ",".join(candidate_item_ids),
        "recommendation_text": prompt
    }
    # 生成完整的推荐文本
    return type, recommendation_text


# 为每个用户生成推荐文本并存储为 JSON 文件
recommendations_forget = []
recommendations_remain = []
for user_id in tqdm(user_profile_df["user_id:token"].unique(),
                    desc=set_color(f"Generating recommendations prompt(top-{CANDIDATE_ITEM_NUM}) ", "pink"),
                    unit="user"):
    type, text = generate_recommendation_text(user_id)
    if type == "remain":
        recommendations_remain.append(text)
    else:
        recommendations_forget.append(text)



# 保存到 JSON 文件
with open(f'{DATASET}_{MODEL}_prompt_top{CANDIDATE_ITEM_NUM}_{selection_strategy}_remain.json', "w", encoding="utf-8") as f:
    json.dump(recommendations_remain, f, ensure_ascii=False, indent=4)

with open(f'{DATASET}_{MODEL}_prompt_top{CANDIDATE_ITEM_NUM}_{selection_strategy}_forget.json', "w", encoding="utf-8") as f:
    json.dump(recommendations_forget, f, ensure_ascii=False, indent=4)