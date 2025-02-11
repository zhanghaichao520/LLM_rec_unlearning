import math
import ast
from tqdm import tqdm
from recbole.utils import set_color
def calculate_hr(ground_truth, predictions):
    hits = 0
    for user, items in predictions.items():
        if len(ground_truth[user]) < 2:
            continue
        if user in ground_truth:
            for item in items:
                if int(float(item)) in ground_truth[user]:
                    hits += 1
                    break  # 一个用户命中一次即可
    return hits / (len(predictions)) if predictions else 0


def ndcg_at_k(ground_truth, predictions, k):
    """
    计算推荐算法的 NDCG@k 指标。

    :param ground_truth: dict, 用户ID作为key，推荐的itemid集合作为value (set)
    :param predictions: dict, 用户ID作为key，topk个推荐的itemid和score的列表作为value (list of tuples)
    :param k: int, 计算NDCG时的k值（即考虑前k个推荐项）
    :return: float, NDCG@k指标
    """

    def dcg_at_k(relevant_items, recommended_items, k):
        """
        计算Discounted Cumulative Gain (DCG)。

        :param relevant_items: set, 用户真实的相关item集合
        :param recommended_items: list of tuples, 推荐的item和推荐分数的列表 (itemid, score)
        :param k: int, 计算DCG时的k值
        :return: float, DCG值
        """
        dcg = 0.0
        for i in range(min(k, len(recommended_items))):
            item_id, score = recommended_items[i]
            # 判断该item是否在真实的相关item集合中
            relevance = 1 if int(float(item_id)) in relevant_items else 0
            dcg += relevance / math.log2(i + 2)  # 使用log2(i+2)来避免log(0)情况
        return dcg

    def ideal_dcg_at_k(relevant_items, k):
        """
        计算Ideal Discounted Cumulative Gain (IDCG)。

        :param relevant_items: set, 用户真实的相关item集合
        :param k: int, 计算IDCG时的k值
        :return: float, IDCG值
        """
        # 根据相关性排序，真实的相关item的IDCG就是将所有相关item按排名顺序进行最大化
        relevant_list = list(relevant_items)
        ideal_recommended_items = [(item_id, 1) for item_id in relevant_list]  # 所有相关项的score设置为1
        return dcg_at_k(relevant_items, ideal_recommended_items, k)

    ndcg_total = 0.0
    user_count = 0

    for user_id, relevant_items in ground_truth.items():
        if user_id in predictions:
            recommended_items = predictions[user_id]
            user_dcg = dcg_at_k(relevant_items, recommended_items, k)
            user_idcg = ideal_dcg_at_k(relevant_items, k)
            # 避免除以0的情况
            if user_idcg > 0:
                user_ndcg = user_dcg / user_idcg
            else:
                user_ndcg = 0.0
            ndcg_total += user_ndcg
            user_count += 1

    # 计算平均NDCG
    return ndcg_total / user_count if user_count > 0 else 0.0


import pandas as pd

def dataframe_to_ground_truth(df):
    """
    将 DataFrame 转换为 ground_truth 格式。

    Args:
        df: 包含 user_id 和 item_id 列的 Pandas DataFrame。

    Returns:
        一个 ground_truth 字典。
        如果输入df为空或者不包含user_id或item_id列，则返回空字典
    """
    if df.empty or 'user_id' not in df.columns or 'item_id' not in df.columns:
        return {}

    ground_truth = {}
    for index, row in df.iterrows():
        user_id = int(row['user_id'])
        item_id = int(row['item_id'])
        if user_id not in ground_truth:
            ground_truth[user_id] = set()
        ground_truth[user_id].add(item_id)
    return ground_truth
import os
from DRAGRU.recbole_utils import RecUtils
MODEL = "LightGCN"
DATASET = "netflix-process"
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
config_files = f"config_file/{DATASET}.yaml"
config = {"normalize_all": False}
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)

rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict=config)
inter_attr_df = rec_utils.ori_testset
# 示例数据 (需要根据你的实际数据进行解析)
ground_truth = dataframe_to_ground_truth(inter_attr_df)

file_path = 'netflix-process_LightGCN_prompt_top50_SelectionStrategy.CA_remain.json_Meta-Llama-3-8B-Instruct_result_llm.csv'
data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

topK=1
predictions_rank = {

}
predictions_score={

}

exception_user = []
for index, row in tqdm(
        data.iterrows(),
        total=len(data),
        ncols=100,
        desc=set_color(f"Parse rec result  ", "pink"),
):
    try:
        user_id = row['user_id']
        # 将字符串转换为字典
        data_dict = ast.literal_eval(row['predict_score'])

        sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)[:topK]

        topK_ori_item_id = [key for key, value in sorted_items]
        topK_ori_item_score = [value for key, value in sorted_items]

        predictions_rank[user_id] = topK_ori_item_id
        predictions_score[user_id] = list(zip(topK_ori_item_id, topK_ori_item_score))


    except Exception as e:
        # print(f"Error processing row {index}: {row['predict_score']}")
        exception_user.append(row['user_id'])

print("HR (rank):", calculate_hr(ground_truth, predictions_rank))
print("NDCG (score):", ndcg_at_k(ground_truth,predictions_score, k=topK))