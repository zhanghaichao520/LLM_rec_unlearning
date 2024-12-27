1. [split_dataset_random.py](split_dataset_random.py): 切割数据集为多个
2. [batch_train.py](batch_train.py) : 批量训练多个数据集， 将模型存储起来
3. [agg_and_eval.py](agg_and_eval.py) : 读取模型目录下的所有模型， 获取最终的metrics

****
1. [train_single_model.py](train_single_model.py) ： 训练单个模型的代码
2. [agg_submodels.py](agg_submodels.py) : 单个模型的评估指标