from SISA.agg_submodels import AggSubmodels
import os
import torch
# 获取candidate item 的传统推荐模型
MODEL = "BPR"
# 处理的数据集
DATASET = "netflix-process-remain"
model_dir = "SISA/saved_BPR_netflix-process-remain/"
# 默认配置文件， 注意 normalize_all: False 便于保留原始的时间和rating
config_files = f"config_file/netflix-process.yaml"
config = {"normalize_all": False,"topk":[5,10,20], "metrics": ["NDCG","Hit"]}
config_file_list = (
    config_files.strip().split(" ") if config_files else None
)

models = []
for filename in os.listdir(model_dir):
    models.append(os.path.join(model_dir, filename))

agg_submodels = AggSubmodels(model=MODEL, dataset=DATASET, config_file_list=config_file_list, config_dict=config)

# 获取所有子模型对每个item的推荐分数
all_batch_scores = []
for model in models:
    batch_scores = agg_submodels.get_recommandation_score(model)
    all_batch_scores.append(batch_scores)


# 初始化一个新的字典来存储结果
average_scores = {}

# 获取所有可能的key（序号），假设每个字典有相同的key
keys = all_batch_scores[0].keys()

# 对每个key，计算所有10个字典中对应tensor的平均值
for key in keys:
    # 提取出所有字典中对应key的tensor，并计算它们的平均值
    tensors = [batch[key] for batch in all_batch_scores]
    # 计算平均值，假设使用torch进行操作
    average_tensor = torch.mean(torch.stack(tensors), dim=0)
    # 将结果存入新字典
    average_scores[key] = average_tensor



result = agg_submodels.get_recommandation_result(models[0], average_scores)
print(result)