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
        assert path.isdir(
            args["backbone_path"]
        ), f"Backbone dir \"{args['backbone_path']}\" doesn't exist."
        preload_backbone = True
        # backbone, _, feature_size = torch.load(
        #     args["backbone_path"], map_location=main_device, weights_only=False
        # )
        backbone, input_img_size, feature_size = load_backbone(backbone_name, pretrain=False)
    else:
        assert "args --backbone-path is empty"
        return 1
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
    dataset_test = data_manager.get_dataset(train=False)
    total_classes = dataset_test.num_classes

    # Select algorithm
    assert args["method"] in ALL_METHODS, f"Unknown method: {args['method']}"
    learner = ALL_METHODS[args["method"]](
        args=args, backbone=backbone, backbone_output=feature_size, data_manager=data_manager, CL_type=CL_type, device=main_device, all_devices=all_gpus
    )

    # Incremental learning
    log_file_path = path.join(args["saving_root"], "test_IL.csv")
    log_file = open(log_file_path, "w", buffering=1)
    print(
        "phase", "acc@avg", "af", file=log_file, sep=","
    )

    print('start incremental learning')
    first_acc = []
    test_subsets = []
    for p in range(args["phases"]):
        test_subsets.append(dataset_test.subset_at_phase(p))
    test_loaders = []
    for p in range(args["phases"]):
        test_loader = make_dataloader(
            test_subsets[p],
            False,
            args["IL_batch_size"],
            args["num_workers"],
            device=main_device,
        )
        test_loaders.append(test_loader)

    # for phase in range(0, args["phases"]):
    for phase in range(0, args["phases"]):
        print(f"Phase {phase}")
        if CL_type == 'CIL':
            total_classes += 1 if phase > 0 else 0

        if CL_type == 'TIL':
            test_loaders = test_loaders[:]
        # else:
        union_test_subset = dataset_test.subset_until_phase(phase)
        union_test_loader = make_dataloader(
            union_test_subset,
            False,
            args["IL_batch_size"],
            args["num_workers"],
            device=main_device,
        )
        
        learner.update_model(phase, total_classes, main_device)
        model_path = os.path.join(args["backbone_path"],f"model_{phase}.pth")
        state_dict = torch.load(
            model_path, map_location=main_device, weights_only=True
        )
        learner.model.load_state_dict(state_dict)
        print(f'loaded from {model_path}')

        # Validation
        if CL_type == 'TIL':
            sum_acc = 0.0
            AF = 0.0
            val_meters = []
            # for p in range(phase + 1):
            for p in range(args["phases"]+1):
                # performance on all previous task
                val_meter = validate(
                    learner.model,
                    test_loaders[p],
                    total_classes,
                    p,
                    desc=f"Phase {p}",
                )
                val_meters.append(val_meter)
                sum_acc += val_meter.accuracy
                if phase > 0 and p < phase:
                    # average forgetting factor
                    AF += val_meter.accuracy - first_acc[p]
                print(
                    f"acc@1: {val_meter.accuracy * 100:.2f}%",
                    f"auc: {val_meter.auc * 100:.2f}%",
                    sep="    ",
                )
                print(
                    phase,
                    f"{val_meter.accuracy*100:.2f}",
                    f"{val_meter.auc*100:.2f}",
                    file=log_file,
                    sep=",",
                )
            # first acc of each task at the last test datasubset
            first_acc.append(val_meter.accuracy)
            AF = AF / phase if phase > 0 else 0.0

            print(
                f"acc@avg: {sum_acc / (phase + 1) * 100:.2f}%",
                f"AF: {AF * 100:.2f}%",
                sep="    ",
            )
            print(
                phase,
                f"{sum_acc / (phase + 1) * 100:.2f}",
                f"{AF * 100:.2f}",
                "\n",
                file=log_file,
                sep=" ",
            )
        # else:

        sum_acc = 0.0
        AF = 0.0

        # # performance on all previous task
        # val_meter = validate(
        #     learner.model,
        #     union_test_loader,  # historical union test set
        #     total_classes,
        #     phase,
        #     desc=f"Phase {phase}",
        # )

        # sum_acc = val_meter.accuracy
        # if CL_type == 'CIL':
        #     AUC = 0
        # else:
        #     AUC = val_meter.auc
        # if phase > 0:
        #     # average forgetting factor
        #     AF = sum_acc - first_acc[0]
        # print(
        #     f"acc@1: {sum_acc * 100:.2f}%",
        #     f"auc: {AUC * 100:.2f}%",
        #     f"AF: {AF * 100:.2f}%",
        #     sep="    ",
        # )
        # print(
        #     phase,
        #     f"{sum_acc * 100:.2f}",
        #     f"{AUC * 100:.2f}",
        #     f"{AF * 100:.2f}",
        #     "\n",
        #     file=log_file,
        #     sep=" ",
        # )
        # # first acc of each task at the last test datasubset
        # first_acc.append(sum_acc)

    log_file.close()

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