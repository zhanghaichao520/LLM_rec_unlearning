import argparse
import sys
import torch.distributed as dist
from logging import getLogger

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

from texttable import Texttable

def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)


    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )
    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"],
    )
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )


    return test_result["hit@10"]  # for the single process


import numpy as np
import csv

if __name__ == "__main__":
    config_dict = {"normalize_all": False, "topk": [10], "metrics": ["Hit", "mrr"]}
    config_files = "config_file/ml-100k.yaml"
    config_file_list = (
        config_files.strip().split(" ") if config_files else None
    )
    # 切割比例
    K = 5
    ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    DATASET = "ml-100k"
    MODEL = "BPR"

    result = np.zeros((K, len(ratios)), dtype=int)

    for index, ratio in enumerate(ratios):
        for label in range(K):
            dataset_name = f"{DATASET}-{int(ratio * 100)}-{label}"

            parser = argparse.ArgumentParser()
            parser.add_argument("--model", "-m", type=str, default=MODEL, help="name of models")
            parser.add_argument(
                "--dataset", "-d", type=str, default=f'{dataset_name}', help="name of datasets"
            )
            parser.add_argument("--config_files", type=str, default=None, help="config files")
            parser.add_argument(
                "--nproc", type=int, default=1, help="the number of process in this group"
            )
            parser.add_argument(
                "--ip", type=str, default="localhost", help="the ip of master node"
            )
            parser.add_argument(
                "--port", type=str, default="5678", help="the port of master node"
            )
            parser.add_argument(
                "--world_size", type=int, default=-1, help="total number of jobs"
            )
            parser.add_argument(
                "--group_offset",
                type=int,
                default=0,
                help="the global rank offset of this group",
            )


            args, _ = parser.parse_known_args()

            res = run_recbole(
                model=args.model,
                dataset=args.dataset,
                config_file_list=config_file_list,
                config_dict=config_dict,
                saved=True,
            )

            result[label, index] = res
            print(f"finished: {dataset_name}, result: {res}")

    print(result)

    # 创建表格对象
    table = Texttable()
    table.set_cols_align(["l"] + ["c"] * 10)  # 第一列左对齐，其余列居中对齐
    table.set_cols_valign(["m"] * 11)  # 所有列都垂直居中

    # 构建表头
    header = [""] + [f"{i * 5}%" for i in range(1, 11)]  # 从5%到50%
    table.add_row(header)

    # 添加矩阵内容到表格中
    for i in range(K):
        row = [f"category{i + 1}"] + list(result[i])  # 每一行的分类加上对应的矩阵值
        table.add_row(row)

    # 打印表格
    print(table.draw())

    # 将矩阵写入 CSV 文件
    with open(f'{DATASET}-{MODEL}-{K}-result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)  # 直接写入矩阵的每一行
