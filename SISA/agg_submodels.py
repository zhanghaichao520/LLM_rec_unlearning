import sys
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
    Interaction
)
from recbole.evaluator import Evaluator, Collector
from recbole.evaluator.collector import DataStruct
from recbole.data.transform import construct_transform
import numpy as np
from tqdm import tqdm
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    get_environment,
    get_flops,
    set_color
)

class AggSubmodels:
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
        self.tot_item_num = self.test_data._dataset.item_num
        self.item_tensor = self.test_data._dataset.get_item_feature().to(self.config["device"])
        self.test_batch_size = self.config["eval_batch_size"]

        # Model loading and initialization
        init_seed(self.config["seed"] + self.config["local_rank"], self.config["reproducibility"])
        self.model = get_model(self.config["model"])(self.config, self.train_data._dataset).to(self.config["device"])
        self.logger.info(self.model)

        transform = construct_transform(self.config)
        flops = get_flops(self.model, self.dataset, self.config["device"], self.logger, transform)
        self.logger.info(set_color("FLOPs", "blue") + f": {flops}")

        # Trainer loading and initialization
        self.trainer = get_trainer(self.config["MODEL_TYPE"], self.config["model"])(self.config, self.model)

        self.data_struct = DataStruct()
        self.evaluator = Evaluator(self.config)

        self.topk = self.config["topk"]


    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(
                Interaction(current_interaction).to(self.config["device"])
            )
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.config["device"]))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.config["device"]).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i


    def get_recommandation_score(self, model_file):
        checkpoint = torch.load(model_file, weights_only=False, map_location=self.config["device"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))
        message_output = "Loading model structure and parameters from {}".format(
            model_file
        )
        self.logger.info(message_output)
        self.model.eval()

        eval_func = self._full_sort_batch_eval

        iter_data = (
            tqdm(
                self.test_data,
                total=len(self.test_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        )

        result = {}
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            iter_data.set_postfix_str(
                set_color("GPU RAM: " + self.get_gpu_usage(self.config["device"]), "yellow")
            )
            result[batch_idx] = scores
        return result


    def get_recommandation_result(self, model_file, batch_scores):
        checkpoint = torch.load(model_file, weights_only=False, map_location=self.config["device"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))
        message_output = "Loading model structure and parameters from {}".format(
            model_file
        )
        self.logger.info(message_output)
        self.model.eval()

        eval_func = self._full_sort_batch_eval

        iter_data = (
            tqdm(
                self.test_data,
                total=len(self.test_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        )

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            iter_data.set_postfix_str(
                set_color("GPU RAM: " + self.get_gpu_usage(self.config["device"]), "yellow")
            )
            scores = batch_scores[batch_idx]

            _, topk_idx = torch.topk(
                scores, max(self.topk), dim=-1
            )  # n_users x k
            pos_matrix = torch.zeros_like(scores, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.data_struct.update_tensor("rec.topk", result)

        result = self.evaluator.evaluate(self.data_struct)
        return result


    def get_gpu_usage(self, device=None):
        r"""Return the reserved memory and total memory of given device in a string.
        Args:
            device: cuda.device. It is the device that the model run on.

        Returns:
            str: it contains the info about reserved memory and total memory of given device.
        """

        reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
        total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

        return "{:.2f} G/{:.2f} G".format(reserved, total)


    def print_env(self):
        environment_tb = get_environment(self.config)
        self.logger.info(
            "The running environment of this training is as follows:\n"
            + environment_tb.draw()
        )