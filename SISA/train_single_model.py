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


def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    model_file=None
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
    test_result = None
    if model_file is None:
        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config["show_progress"]
        )
        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=config["show_progress"]
        )
    else:
        # model evaluation
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=config["show_progress"],
            model_file=model_file
        )
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )


    return test_result["hit@10"]  # for the single process



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k-SISA-part0", help="name of datasets"
    )
    parser.add_argument(
        "--model_file", "-mf", type=str, default=None, help="model_file"
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

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    config_dict = {"normalize_all": False, "topk": [10],
                   "metrics": ["Hit", "mrr"],
                   "checkpoint_dir": f"SISA/saved_{args.model}_{args.dataset.split('-SISA')[0]}"}

    res = run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
        saved=True,
        model_file=args.model_file
    )
    print(res)
