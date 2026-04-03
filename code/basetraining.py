# -*- coding: utf-8 -*-

import torch
from os import path
from tqdm import tqdm
from config import load_args, ALL_METHODS
from models import load_backbone
from typing import Any, Dict, List, Tuple, Optional
from datasets import Features, load_dataset, DataManager
from utils import set_determinism, validate
from torch._prims_common import DeviceLikeType
from torch.utils.data import Dataset, DataLoader

import warnings
import copy
from utils.visual import incremental_features

warnings.filterwarnings("ignore", category=UserWarning)

def make_dataloader(
    dataset: Dataset,
    shuffle: bool = False,
    batch_size: int = 256,
    num_workers: int = 8,
    device: Optional[DeviceLikeType] = None,
    persistent_workers: bool = False,
) -> DataLoader:
    pin_memory = (device is not None) and (torch.device(device).type == "cuda")
    config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "pin_memory_device": str(device) if pin_memory else "",
        "persistent_workers": persistent_workers,
        "drop_last": True,
    }
    try:
        from prefetch_generator import BackgroundGenerator

        class DataLoaderX(DataLoader):
            def __iter__(self):
                return BackgroundGenerator(super().__iter__())

        return DataLoaderX(dataset, **config)
    except ImportError:
        return DataLoader(dataset, **config)


def main(args: Dict[str, Any]):
    backbone_name = args["backbone"]
    cache_path = '/home/yejianbin/CIL/ACIL/figures'

    # Select device
    if args["cpu_only"] or not torch.cuda.is_available():
        main_device = torch.device("cpu")
        all_gpus = None
    elif args["gpus"] is not None:
        gpus = args["gpus"]
        main_device = torch.device(f"cuda:{gpus[0]}")
        all_gpus = [torch.device(f"cuda:{gpu}") for gpu in gpus]
    else:
        main_device = torch.device("cuda:0")
        all_gpus = None

    if args["seed"] is not None:
        set_determinism(args["seed"])

    if args["backbone_path"] is not None:
        assert path.isfile(
            args["backbone_path"]
        ), f"Backbone file \"{args['backbone_path']}\" doesn't exist."
        preload_backbone = True
        # backbone, _, feature_size = torch.load(
        #     args["backbone_path"], map_location=main_device, weights_only=False
        # )
        backbone, input_img_size, feature_size = load_backbone(backbone_name, pretrain=False)
        state_dict = torch.load(
            args["backbone_path"], map_location=main_device, weights_only=True
        )
        # backbone.load_state_dict(state_dict)
        # print(f'loaded from {args["backbone_path"]}')
    else:
        # Load model pre-train on ImageNet if there is no base training dataset.
        preload_backbone = False
        load_pretrain = args["base_ratio"] == 0 or "ImageNet" not in args["dataset"]
        backbone, input_img_size, feature_size = load_backbone(backbone_name, pretrain=load_pretrain)
        if load_pretrain:
            assert args["dataset"] != "ImageNet", "Data may leak!!!"
    backbone = backbone.to(main_device, non_blocking=True)
    CL_type = args["CL_type"]

    csv_dir = args['csv_dir']
    dataset_args = {
        "name": args["dataset"],
        "root": args["data_root"],
        "base_ratio": args["base_ratio"],
        "num_phases": args["phases"],
        "shuffle_seed": args["dataset_seed"] if "dataset_seed" in args else None,
        'image_size': 128,
        # 'num_classes': 4,
        'num_classes': 2,
        'csv_dir': csv_dir,
        'IL_batch_size':  args["batch_size"],
        'num_workers':  args["num_workers"],
        'reverse': True if args["reverse"] else False,
        'CL_type': CL_type,
    }
    data_manager = DataManager(**dataset_args)
    dataset_train = data_manager.get_dataset(train=True)
    dataset_test = data_manager.get_dataset(train=False)

    # Base training
    for phase in range(args["phases"]):
        train_subset = dataset_train.subset_at_phase(phase)
        test_subset = dataset_test.subset_at_phase(phase)
        train_loader = make_dataloader(
            train_subset,
            True,
            args["batch_size"],
            args["num_workers"],
            device=main_device,
        )
        test_loader = make_dataloader(
            test_subset,
            False,
            args["batch_size"],
            args["num_workers"],
            device=main_device,
        )
        total_classes = dataset_train.num_classes

        backbone, input_img_size, feature_size = load_backbone(backbone_name, pretrain=load_pretrain)
        backbone = backbone.to(main_device, non_blocking=True)
        learner = ALL_METHODS[args["method"]](
            args=args, backbone=backbone, backbone_output=feature_size, data_manager=data_manager, CL_type=CL_type, device=main_device, all_devices=all_gpus
        )
        learner.base_training(
            train_loader,
            test_loader,
            # dataset_train.base_size,
            dataset_train.num_classes,
        )

import random
import numpy as np
import torch
import os

def set_seed(seed):
    # 设置Python内置random模块的随机种子
    random.seed(seed)
    # 设置numpy的随机种子
    np.random.seed(seed)
    # 如果使用PyTorch，还需要设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 如果你在GPU上运行PyTorch，还需要设置这个
    torch.cuda.manual_seed_all(seed)
    # 设置环境变量CUBLAS_WORKSPACE_CONFIG以确保卷积操作的可重复性
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # 设置PyTorch的确定性操作标志
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    import time
    start_time = time.time()
    
    set_seed(0)
    main(load_args())

    end_time = time.time()
    print("Time used:", end_time - start_time)
    '''
    python main.py Finetune --dataset MDCDDataset --base-ratio 0.17 --phases 6 --CL-type TIL \
         --csv-dir /mnt/200ssddata2t/yejianbin/MDCD-DB/Sequetail_incremental_DB/Task_incremental_DB/cross_dataset \
         --batch-size 16 --num-workers 8 --backbone GAT_video_audio \
        --base-epochs 5 --learning-rate 0.001 --gpus 1
    '''