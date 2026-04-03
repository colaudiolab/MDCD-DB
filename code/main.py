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


def check_cache_features(root: str) -> bool:
    files_list = ["X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt"]
    for file in files_list:
        if not path.isfile(path.join(root, file)):
            return False
    return True


@torch.no_grad()
def cache_features(
    backbone: torch.nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: Optional[DeviceLikeType] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    X_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    for X, y in tqdm(dataloader, "Caching"):
        # X: torch.Tensor = backbone(X.to(device))
        video, audio = X
        video = video.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        video_out, audio_out, X = backbone(video, audio)
        y: torch.Tensor = y.to(torch.int16, non_blocking=True)
        X_all.append(X.cpu())
        y_all.append(y.cpu())
    return torch.cat(X_all), torch.cat(y_all)


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

    if args["dataset"] == "FakeAVCeleb":
        csv_dir = '/mnt/200ssddata2t/yejianbin/FakeAVCeleb/csv_deepfake_method'
    elif args["dataset"] == "multidataset":
        csv_dir = '/mnt/200ssddata2t/yejianbin/multidataset'
    elif args["dataset"] == "multidataset_imbalance_1_5":
        csv_dir = '/mnt/200ssddata2t/yejianbin/multidataset/imbalance_1_5'
    elif args["dataset"] == "multidataset_imbalance_1_10":
        csv_dir = '/mnt/200ssddata2t/yejianbin/multidataset/imbalance_1_10'
    elif args["dataset"] == "multidataset_imbalance_1_20":
        csv_dir = '/mnt/200ssddata2t/yejianbin/multidataset/imbalance_1_20'
    elif args["dataset"] == "multidataset_imbalance_1_40":
        csv_dir = '/mnt/200ssddata2t/yejianbin/multidataset/imbalance_1_40'
    elif args["dataset"] == "multidataset_imbalance_2_5_10_20":
        csv_dir = '/mnt/200ssddata2t/yejianbin/multidataset/imbalance_2_5_10_20'
    elif args["dataset"] == "multidataset_imbalance_2_5_10_20_re":
        csv_dir = '/mnt/200ssddata2t/yejianbin/multidataset/imbalance_2_5_10_20_re'
    elif args["dataset"] == "MDCDDataset":
        if CL_type == 'CIL':
            csv_dir = '/mnt/200ssddata2t/yejianbin/MDCD-DB/Class_incremental_FakeAVCeleb'
        elif CL_type == 'DIL':
            csv_dir = '/mnt/200ssddata2t/yejianbin/MDCD-DB/Sequetail_incremental_DB/Domain_incremental_DB'
        elif CL_type == 'TIL':
            csv_dir = '/mnt/200ssddata2t/yejianbin/MDCD-DB/Sequetail_incremental_DB/Task_incremental_DB'
        csv_dir = args['csv_dir']
    else:
        csv_dir = None
        assert args["dataset"] in ["FakeAVCeleb", "multidataset", "multidataset_imbalance_1_5"], "Unknown dataset"
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
    # dataset_train = load_dataset(train=True, augment=False, **dataset_args)
    # dataset_test = load_dataset(train=False, augment=False, **dataset_args)
    dataset_train = data_manager.get_dataset(train=True)
    dataset_test = data_manager.get_dataset(train=False)

    # Select algorithm
    assert args["method"] in ALL_METHODS, f"Unknown method: {args['method']}"
    learner = ALL_METHODS[args["method"]](
        args=args, backbone=backbone, backbone_output=feature_size, data_manager=data_manager, CL_type=CL_type, device=main_device, all_devices=all_gpus
    )

    # Base training
    if args["base_ratio"] > 0:
        train_subset = dataset_train.subset_at_phase(0)
        test_subset = dataset_test.subset_at_phase(0)
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
        if preload_backbone:
            learner.load_model(total_classes, state_dict, train_loader)
        else:
            learner.base_training(
                train_loader,
                test_loader,
                # dataset_train.base_size,
                dataset_train.num_classes,
            )
        # incremental_features(cache_path, learner.backbone, test_loader, main_device, phase=0)

    # Incremental learning
    log_file_path = path.join(args["saving_root"], "IL.csv")
    log_file = open(log_file_path, "w", buffering=1)
    print(
        "phase", "acc@avg", "af", file=log_file, sep=","
    )

    print('start incremental learning')
    first_acc = []
    for phase in range(0, args["phases"]):
        print(f"Phase {phase}")
        if CL_type == 'CIL':
            total_classes += 1 if phase > 0 else 0
        incremental_dataset_train = copy.deepcopy(dataset_train)

        if CL_type == 'TIL':
            test_subsets = []
            for p in range(phase + 1):
                test_subsets.append(dataset_test.subset_at_phase(p))
            test_loaders = []
            for p in range(phase + 1):
                test_loader = make_dataloader(
                    test_subsets[p],
                    False,
                    args["IL_batch_size"],
                    args["num_workers"],
                    device=main_device,
                )
                test_loaders.append(test_loader)
        # else:
        union_test_subset = dataset_test.subset_until_phase(phase)
        union_test_loader = make_dataloader(
            union_test_subset,
            False,
            args["IL_batch_size"],
            args["num_workers"],
            device=main_device,
        )
        
        if phase == 0:
            learner.learn(incremental_dataset_train, dataset_train.base_size, phase, total_classes, "Re-align")
        else:
            learner.learn(incremental_dataset_train, dataset_train.phase_size, phase, total_classes)
            # incremental_features(cache_path, learner.backbone, test_loaders[0], main_device, phase)
        learner.before_validation(phase)

        # Validation
        if CL_type == 'TIL':
            sum_acc = 0.0
            AF = 0.0
            val_meters = []
            for p in range(phase + 1):
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

        # performance on all previous task
        val_meter = validate(
            learner.model,
            union_test_loader,  # historical union test set
            total_classes,
            phase,
            desc=f"Phase {phase}",
        )

        sum_acc = val_meter.accuracy
        if CL_type == 'CIL':
            AUC = 0
        else:
            AUC = val_meter.auc
        if phase > 0:
            # average forgetting factor
            AF = sum_acc - first_acc[0]
        print(
            f"acc@1: {sum_acc * 100:.2f}%",
            f"auc: {AUC * 100:.2f}%",
            f"AF: {AF * 100:.2f}%",
            sep="    ",
        )
        print(
            phase,
            f"{sum_acc * 100:.2f}",
            f"{AUC * 100:.2f}",
            f"{AF * 100:.2f}",
            "\n",
            file=log_file,
            sep=" ",
        )
        # first acc of each task at the last test datasubset
        first_acc.append(sum_acc)

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
    '''
    python main.py ACIL --dataset FakeAVCeleb --base-ratio 0.2 --phases 3 \
    --data-root /mnt/200ssddata2t/yejianbin/FakeAVCeleb/cropped_faces --batch-size 64 --num-workers 16 --backbone MRDF_CE \
    --label-smoothing 0 --base-epochs 30 --learning-rate 0.001  \
    --buffer-size 64 --IL-batch-size 64 --gpus 1 0

    python main.py ACIL --dataset FakeAVCeleb --base-ratio 0.2 --phases 6 \
    --data-root /mnt/200ssddata2t/yejianbin/FakeAVCeleb/cropped_faces --batch-size 16 --num-workers 16 --backbone GAT_video_audio \
    --label-smoothing 0 --base-epochs 4 --learning-rate 0.001  \
    --buffer-size 64 --IL-batch-size 16 --gpus 3 2
    '''