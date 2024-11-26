import sys
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
    Interaction
)
import pandas as pd
from recbole.data.transform import construct_transform
import numpy as np
from recbole.model.general_recommender.dmf import DMF

from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    get_environment,
    get_flops,
    set_color
)

class RecUtils:
    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        self.config = Config(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
        )
        init_seed(self.config["seed"], self.config["reproducibility"])
        init_logger(self.config)
        self.logger = getLogger()
        self.logger.info(sys.argv)
        self.logger.info(self.config)
        # Dataset filtering and processing
        self.dataset = create_dataset(self.config)

        # Dataset splitting
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)
        self.ori_trainset = self._get_ori_trainset()
        self.ori_testset = self._get_ori_testset()

        # Model loading and initialization
        init_seed(self.config["seed"] + self.config["local_rank"], self.config["reproducibility"])
        self.model = get_model(self.config["model"])(self.config, self.train_data._dataset).to(self.config["device"])
        self.logger.info(self.model)

        transform = construct_transform(self.config)
        flops = get_flops(self.model, self.dataset, self.config["device"], self.logger, transform)
        self.logger.info(set_color("FLOPs", "blue") + f": {flops}")

        # Trainer loading and initialization
        self.trainer = get_trainer(self.config["MODEL_TYPE"], self.config["model"])(self.config, self.model)

        self.user_id_token2his_idx_cache = self._initialize_cache()  # 用于缓存 user_token_id -> his_idx 映射

    def _initialize_cache(self):
        """
        遍历 test_data，构建 user_token_id 到 his_idx 的映射。
        """
        cache = {}
        for batch_idx, batched_data in enumerate(self.test_data):
            inter, history_index, _, _ = batched_data

            # 遍历当前 batch 的所有用户 ID
            for idx in range(len(inter[self.dataset.uid_field])):
                user_id_token = inter[self.dataset.uid_field][idx].item()
                if user_id_token not in cache:
                    idx_row = history_index[0]
                    idx_col = history_index[1]

                    # 筛选 idx_row=0 的位置
                    mask = (idx_row == idx)
                    filtered_col_indices = idx_col[mask]  # 保留 idx_row=0 的列
                    # 直接获取对应的 history_index
                    cache[user_id_token] = filtered_col_indices

        return cache

    def train(self):
        best_valid_score, best_valid_result = self.trainer.fit(
            self.train_data, self.valid_data, saved=True, show_progress=self.config["show_progress"]
        )

        # model evaluation
        test_result = self.trainer.evaluate(
            self.test_data, load_best_model=True, show_progress=self.config["show_progress"]
        )

        self.print_env()

        self.logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
        self.logger.info(set_color("test result", "yellow") + f": {test_result}")

        return self.trainer.saved_model_file


    def get_ori_item_id(self, token_item_id):
        r"""
            原始id是字符串
            编码后的token是int
        """
        return self.dataset.field2id_token[self.dataset.iid_field][token_item_id]

    def get_ori_user_id(self, token_user_id):
        return self.dataset.field2id_token[self.dataset.uid_field][token_user_id]

    def get_encode_item_token(self, ori_item_id):
        return self.dataset.field2token_id[self.dataset.iid_field][str(ori_item_id)]

    def get_encode_user_token(self, ori_user_id):
        return self.dataset.field2token_id[self.dataset.uid_field][str(ori_user_id)]


    def _get_recommandation_list(self, user_id_token, topk, model_file):
        checkpoint = torch.load(model_file, weights_only=False, map_location=self.config["device"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        self.model.eval()
        interaction = Interaction({self.dataset.uid_field: torch.tensor([user_id_token])}).to(self.config["device"])

        scores = self.model.full_sort_predict(interaction)
        scores[0] = -np.inf

        his_idx = self.user_id_token2his_idx_cache.get(user_id_token, None)

        if his_idx is not None:
            scores[his_idx] = -np.inf

        _, result_tensor = torch.topk(scores, topk, dim=-1)
        return result_tensor

    def get_recommandation_list(self, ori_user_id, topk, model_file):
        """
            传入原始用户id
            返回推荐的原始itemid
        """

        user_id_token = self.get_encode_user_token(ori_user_id)

        item_token_list = self._get_recommandation_list(user_id_token, topk, model_file)

        rec_ori_item_id_result = []
        for item_token in item_token_list:
            rec_ori_item_id_result.append(self.get_ori_item_id(item_token.item()))

        return rec_ori_item_id_result

    def _get_ori_trainset(self):
        # 定义空列表用于存储每一行数据
        data = []

        for row in self.train_data.dataset:
            user_id = self.get_ori_user_id(row[self.dataset.uid_field].item())
            item_id = self.get_ori_item_id(row[self.dataset.iid_field].item())
            rating = int(row["rating"].item())
            data.append([user_id, item_id, rating])

        # 定义列名
        columns = ['user_id', 'item_id', 'rating']

        # 将列表转换为 DataFrame
        return pd.DataFrame(data, columns=columns)

    def _get_ori_testset(self):
        # 定义空列表用于存储每一行数据
        data = []

        for row in self.test_data.dataset:
            user_id = self.get_ori_user_id(row[self.dataset.uid_field].item())
            item_id = self.get_ori_item_id(row[self.dataset.iid_field].item())
            rating = int(row["rating"].item())
            data.append([user_id, item_id, rating])

        # 定义列名
        columns = ['user_id', 'item_id', 'rating']

        # 将列表转换为 DataFrame
        return pd.DataFrame(data, columns=columns)

    def print_env(self):
        environment_tb = get_environment(self.config)
        self.logger.info(
            "The running environment of this training is as follows:\n"
            + environment_tb.draw()
        )