# -*- coding: utf-8 -*-
"""
Implementation of the ACIL [1] and the G-ACIL [2].
The G-ACIL is a generalization of the ACIL in the generalized setting.
For the popular setting, the G-ACIL is equivalent to the ACIL.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
"""

import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from .Buffer import RandomBuffer
from torch.nn import DataParallel
from .Learner import Learner, loader_t
from .AnalyticLinear import AnalyticLinear, RecursiveLinear
import copy
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, nb_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.nb_classes = nb_classes
        self.fc = torch.nn.Linear(backbone_output, nb_classes)
    def forward(self, video, audio, train=True, phase=0):
        out = self.backbone(video, audio)
        video_out, audio_out, x = out['video'], out['audio'], out['features']
        x = self.fc(x)
        outputs = {'logits':x, 'video':video_out, 'audio':audio_out}
        return outputs

class ACIL(torch.nn.Module):
    def __init__(
        self,
        backbone_output: int,
        backbone: torch.nn.Module = torch.nn.Flatten(),
        buffer_size: int = 8192,
        gamma: float = 1e-3,
        device=None,
        dtype=torch.double,
        linear: type[AnalyticLinear] = RecursiveLinear,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        self.eval()

    @torch.no_grad()
    def feature_expansion(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(video, audio)
        video_out, audio_out, x = outputs['video'], outputs['audio'], outputs['features']
        return self.buffer(x)
        # return x

    @torch.no_grad()
    def forward(self, video: torch.Tensor, audio: torch.Tensor, train=True, phase: int=0) -> torch.Tensor:
        logits = self.analytic_linear(self.feature_expansion(video, audio))
        outputs = {'logits':logits}
        return outputs

    @torch.no_grad()
    def fit(self, video: torch.Tensor, audio: torch.Tensor, y: torch.Tensor, phase: int = 0, *args, **kwargs) -> None:
        Y = torch.nn.functional.one_hot(y)
        X = self.feature_expansion(video, audio)
        self.analytic_linear.fit(X, Y)

    @torch.no_grad()
    def update(self) -> None:
        self.analytic_linear.update()


class ACILLearner(Learner):
    """
    This implementation is for the G-ACIL [2], a general version of the ACIL [1] that
    supports mini-batch learning and the general CIL setting.
    In the traditional CIL settings, the G-ACIL is equivalent to the ACIL.
    """

    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        data_manager, 
        CL_type: str,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        super().__init__(args, backbone, backbone_output, data_manager, device, all_devices)
        self.learning_rate: float = args["learning_rate"]
        self.buffer_size: int = args["buffer_size"]
        self.gamma: float = args["gamma"]
        self.base_epochs: int = args["base_epochs"]
        self.warmup_epochs: int = args["warmup_epochs"]
        self.make_model()
        self.CL_type: str = CL_type
        self.nb_classes: int = 0


    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model = self.wrap_data_parallel(model)

        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()
        # optimizer = torch.optim.SGD(
        #     params,
        #     lr=self.learning_rate,
        #     momentum=self.args["momentum"],
        #     weight_decay=self.args["weight_decay"],
        # )
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.base_epochs - self.warmup_epochs, eta_min=1e-6 # type: ignore
        # )
        # if self.warmup_epochs > 0:
        #     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #         optimizer,
        #         start_factor=1e-3,
        #         total_iters=self.warmup_epochs,
        #     )
        #     scheduler = torch.optim.lr_scheduler.SequentialLR(
        #         optimizer, [warmup_scheduler, scheduler], [self.warmup_epochs]
        #     )

        # criterion = torch.nn.CrossEntropyLoss(
        #     label_smoothing=self.args["label_smoothing"]
        # ).to(self.device, non_blocking=True)
        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        best_acc = 0.0
        logging_file_path = path.join(self.args["saving_root"], "base_training.csv")
        logging_file = open(logging_file_path, "w", buffering=1)
        print(
            "epoch",
            "best_acc@1",
            "loss",
            "acc@1",
            "acc@5",
            "f1-micro",
            "training_loss",
            "training_acc@1",
            "training_acc@5",
            "training_f1-micro",
            "training_learning-rate",
            file=logging_file,
            sep=",",
        )

        for epoch in range(self.base_epochs + 1):
            if epoch != 0:
                print(
                    f"Base Training - Epoch {epoch}/{self.base_epochs}",
                    f"(Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']})",
                )
                model.train()
                for indecs, X, y,_ in tqdm(train_loader, "Training"):
                    video, audio = X
                    video = video.to(self.device, non_blocking=True)
                    audio = audio.to(self.device, non_blocking=True)
                    # X: torch.Tensor = X.to(self.device, non_blocking=True)
                    video_label, audio_label, y = y
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                    audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    outs = model(video, audio)
                    video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
                    # loss1 = criterion(video_out, video_label)
                    # loss2 = criterion(audio_out, audio_label)
                    # loss3 = criterion(logits, y)
                    # loss = loss1 + loss2 + loss3
                    loss: torch.Tensor = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    break
                scheduler.step()

            # Validation on training set
            model.eval()
            # train_meter = validate(
            #     model, train_loader, baseset_size, desc="Training (Validation)"
            # )
            # print(
            #     f"loss: {train_meter.loss:.4f}",
            #     f"acc@1: {train_meter.accuracy * 100:.3f}%",
            #     f"auc: {train_meter.auc * 100:.3f}%",
            #     f"f1-micro: {train_meter.f1_micro * 100:.3f}%",
            #     sep="    ",
            # )

            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                self.save_object(
                    # (self.backbone, X.shape[1], self.backbone_output),
                    # self.backbone.state_dict(),
                    # "backbone.pth",
                    model.state_dict(),
                    "model.pth",
                )

            # Validation on testing set
            print(
                f"loss: {val_meter.loss:.4f}",
                f"acc@1: {val_meter.accuracy * 100:.3f}%",
                f"auc: {val_meter.auc * 100:.3f}%",
                f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
                f"best_acc@1: {best_acc * 100:.3f}%",
                sep="    ",
            )
            print(
                epoch,
                best_acc,
                val_meter.loss,
                val_meter.accuracy,
                val_meter.auc,
                val_meter.f1_micro,
                # train_meter.loss,
                # train_meter.accuracy,
                # train_meter.auc,
                # train_meter.f1_micro,
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
        logging_file.close()
        self.model = self.load_object(model, "model.pth")
        self.backbone = self.model.backbone
        self.backbone.eval()
        self.make_model()


    def make_model(self) -> None:
        self.model = ACIL(
            self.backbone_output,
            self.wrap_data_parallel(self.backbone),
            self.buffer_size,
            self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
        )

    # @torch.no_grad()
    def learn(
        self,
        # data_loader: loader_t,
        dataset,
        incremental_size: int,
        phase: int,
        nb_classes: int,
        desc: str = "Incremental Learning",
    ) -> None:
        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        dataset = dataset.subset_at_phase(phase, self.memory)
        data_loader = DataLoader(
            dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )

        self.model.eval()
        for indecs, X, y,_ in tqdm(data_loader, desc=desc):
            video, audio = X
            video = video.to(self.device, non_blocking=True)
            audio = audio.to(self.device, non_blocking=True)
            video_label, audio_label, y = y
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            self.model.fit(video, audio, y, increase_size=incremental_size)
            break
    
    def before_validation(self, phase: int) -> None:
        self.model.update()
        if phase > 0:
            self.save_object(
                self.model.state_dict(),
                f"model_{phase}.pth"
            )

    def inference(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.model(video, audio)

    @torch.no_grad()
    def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.all_devices is not None and len(self.all_devices) > 1:
            return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
        return model

    def load_model(self, baseset_size, state_dict, data_loader):
        print('loading pretrained model')
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model.load_state_dict(state_dict)
        model = self.wrap_data_parallel(model)
        self.backbone = model.backbone

        self.backbone.eval()
        self.make_model()