from tqdm import tqdm
from recbole.utils import set_color
import pandas as pd
import torch
import ast
from DRAGRU.recbole_utils import RecUtils
# 读取 CSV 文件，注意分隔符为制表符
file_path_ori = 'ml-1m_BPR_prompt_top50_remain.json_Meta-Llama-3-8B-Instruct_result_ori.csv'
file_path_llm = 'ml-1m_BPR_prompt_top50_remain.json_Meta-Llama-3-8B-Instruct_result_llm.csv'

# file_path_ori = 'ml-100k_LightGCN_prompt_top50_forget.json_Meta-Llama-3-8B-Instruct_result_ori.csv'
# file_path_llm = 'ml-100k_LightGCN_prompt_top50_forget.json_Meta-Llama-3-8B-Instruct_result_llm.csv'

# file_path_ori = 'ml-100k_LightGCN_prompt_top50_remain.json_Meta-Llama-3-8B-Instruct_result_ori.csv'
# file_path_llm = 'ml-100k_LightGCN_prompt_top50_remain.json_Meta-Llama-3-8B-Instruct_result_llm.csv'

file_path_list = [file_path_ori,file_path_llm]
# 获取candidate item 的传统推荐模型
MODEL = "BPR"
# 处理的数据集
DATASET = "ml-1m"
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
topK = [5,10,20]
config_files = f"config_file/{DATASET}.yaml"
config = {"normalize_all": False, "topk": topK}
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)

rec_utils = RecUtils(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict = config)
def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)


for file_path in file_path_list:
    data = pd.read_csv(file_path, sep='\t', encoding='utf-8')

    # 初始化张量
    topk_idx = torch.zeros((rec_utils.dataset.user_num-1, max(topK)), dtype=torch.int64)

    exception_user = []
    for index, row in tqdm(
        data.iterrows(),
        total=len(data),
        ncols=100,
        desc=set_color(f"Parse rec result  ", "pink"),
    ):
        try:
            encode_user_id = rec_utils.get_encode_user_token(row['user_id'])
            # 将字符串转换为字典
            data_dict = ast.literal_eval(row['predict_score'])

            # 按照值从大到小排序，并取出前10个键
            topK_ori_item_id = [key for key, value in sorted(data_dict.items(), key=lambda item: item[1], reverse=True)[:max(topK)]]

            topK_encode_item_id = []
            for ori_item_id in topK_ori_item_id:
                encode_item_id = rec_utils.get_encode_item_token(ori_item_id)
                topK_encode_item_id.append(encode_item_id)
            topk_idx[encode_user_id - 1] = torch.tensor(topK_encode_item_id, dtype=torch.int64)

        except Exception as e:
            # print(f"Error processing row {index}: {row['predict_score']}")
            exception_user.append(row['user_id'])


    pos_matrix = torch.zeros((rec_utils.dataset.user_num-1, rec_utils.dataset.item_num), dtype=torch.int64)

    iter_data = (
        tqdm(
            rec_utils.test_data,
            total=len(rec_utils.test_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
        )
    )
    row_idx = 0
    for batch_idx, batched_data in enumerate(iter_data):
        interaction, _, positive_u, positive_i = batched_data
        pos_matrix[row_idx+positive_u, positive_i] = 1
        row_idx = row_idx + torch.unique(positive_u).numel()


    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
    result = torch.cat((pos_idx, pos_len_list), dim=1)

    print(f"result.shape : {result.shape}")
    print(f"exception_user: {exception_user}")

    from recbole.evaluator.collector import DataStruct
    data_struct = DataStruct()
    data_struct.update_tensor("rec.topk", result)

    from recbole.evaluator.metrics import *

    print(f"{file_path} result: ")

    hit = Hit(rec_utils.config)
    metric_val = hit.calculate_metric(data_struct)
    print(metric_val)

    ndcg = NDCG(rec_utils.config)
    metric_val = ndcg.calculate_metric(data_struct)
    print(metric_val)

    print("\n")


