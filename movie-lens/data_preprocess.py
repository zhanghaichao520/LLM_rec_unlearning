from LLM_rec_unlearning.recbole_utils import RecUtils
import pandas as pd
import os
import json
from tqdm import tqdm
from recbole.utils import set_color

def find_model_files(directory_path, model_name):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件名是否包含 "abc"
        if model_name in filename:
            return os.path.join(directory_path, filename)

    return None

CANDIDATE_ITEM_NUM = 50
HISTORY_INTER_NUM = 100
# 获取candidate item 的传统推荐模型
MODEL = "LightGCN"
# 处理的数据集
DATASET = "ml-1m"
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
config = {"normalize_all": False}
# config_files = None
config_files = "config_file/ml-1m.yaml"
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)

rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict = config)
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
inter_df = rec_utils.ori_trainset

# 创建电影属性字典，方便通过 item_id 查找电影信息
item_attr_dict = {
    str(row["item_id:token"]): {
        "movie_title": row["movie_title:token_seq"],
        "release_year": row["release_year:token"],
        "class": row["class:token_seq"]
    }
    for _, row in item_attr_df.iterrows()
}

# 生成推荐文本的函数
def generate_recommendation_text(user_id):
    # 获取用户的个人信息
    user_info = user_profile_df[user_profile_df['user_id:token'] == user_id].iloc[0]
    user_profile_text = f"He is {user_info['age:token']} years old, and works as {user_info['occupation:token']}."

    # 从训练集中获取用户的历史观看记录
    user_history = inter_df[inter_df['user_id'] == str(user_id)].head(HISTORY_INTER_NUM)
    history_movies = []
    for _, row in user_history.iterrows():
        item_info = item_attr_dict.get(str(row["item_id"]))
        if item_info:
            history_movies.append(
                f"a {item_info['class']} movie called {item_info['movie_title']} ({item_info['release_year']}), and scored it {20 * int(row['rating'])}.")

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

    return recommendation_text


# 为每个用户生成推荐文本并存储为 JSON 文件
recommendations = []
for user_id in tqdm(user_profile_df["user_id:token"].unique(),
                    desc=set_color(f"Generating recommendations prompt(top-{CANDIDATE_ITEM_NUM}) ", "pink"),
                    unit="user"):
    recommendations.append(generate_recommendation_text(user_id))


# 保存到 JSON 文件
with open(f'{DATASET}_{MODEL}_prompt_top{CANDIDATE_ITEM_NUM}.json', "w", encoding="utf-8") as f:
    json.dump(recommendations, f, ensure_ascii=False, indent=4)