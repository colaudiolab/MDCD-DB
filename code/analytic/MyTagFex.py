# -*- coding: utf-8 -*-
"""
Implementation of the TagFex
"""

import torch
import torch.nn as nn
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
from .Learner import Learner, loader_t
from torch.utils.data import DataLoader
from copy import deepcopy

class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, baseset_size):
        super().__init__()
        self.raw_backbone = deepcopy(backbone)
        self.ta_net = backbone
        # self.fc = torch.nn.Linear(backbone_output, baseset_size)
        self.ts_nets = nn.ModuleList()
        self.out_dim = backbone_output
        self.classifier = None
        self.ts_attn = None
        self.trans_classifier = None
        self.aux_classifier = None

        # contrastive learning projector
        proj_hidden_dim = 2048
        proj_output_dim = 1024
        self.ta_feature_dim = backbone.fmap_dim
        self.projector = nn.Sequential(
            SimpleLinear(self.ta_feature_dim, proj_hidden_dim),
            nn.ReLU(True),
            SimpleLinear(proj_hidden_dim, proj_output_dim),
        )
        self.predictor = None

    def forward(self, video, audio, phase=0, train=True):
        ts_outs = [net(video, audio) for net in self.ts_nets]
        ts_features = [out['features'] for out in ts_outs]
        ts_features = torch.cat(ts_features, dim=-1)
        logits = self.classifier(ts_features)
        outputs = {
            'logits': logits,
            'ts_features': ts_features
        }

        if train:
            teacher_outs = self.ta_net(video, audio)
            teacher_fmap = teacher_outs['fmaps']    # (bs, H*W, C) 
            embedding = self.projector(teacher_fmap.mean(1))
            outputs.update({
                    'ta_feature': teacher_fmap,
                    'embedding': embedding,
            })
            if self.aux_classifier is not None:
                aux_logits = self.aux_classifier(ts_features[:,-self.out_dim:])
                outputs.update(aux_logits=aux_logits)

            if self.trans_classifier is not None:
                ts_feature = ts_outs[-1]["fmaps"]
                ta_features = teacher_fmap  
                merged_feature = self.ts_attn(ta_features.detach(), ts_feature).mean(1)
                trans_logits = self.trans_classifier(merged_feature)
                outputs.update(trans_logits=trans_logits)

            if self.predictor is not None:
                predicted_feature = self.predictor(teacher_fmap.mean(1))
                outputs.update(predicted_feature=predicted_feature)

        # return (logits, aux_logits, trans_logits, embedding, predicted_feature)
        return outputs
    
    def update_network(self, num_new_classes, device) -> None:
        new_ts_net = deepcopy(self.raw_backbone)
        new_ts_net.to(device)
        self.ts_nets.append(new_ts_net)
        if len(self.ts_nets) > 1:
            # init from interpolation
            gamma = 0.95
            for p_ta, p_ts_old, p_ts_new in zip(self.ta_net.parameters(), self.ts_nets[-2].parameters(), self.ts_nets[-1].parameters()):
                p_ts_new.data = gamma * p_ts_old.data + (1 - gamma) * p_ta.data
        
        new_dim = new_ts_net.out_dim
        self.feature_dim = sum(net.out_dim for net in self.ts_nets)
        classifier = SimpleLinear(self.feature_dim, num_new_classes, device=device)
        if self.classifier is not None:
            nb_output = self.classifier.out_features
            classifier.weight.data[:nb_output, : self.feature_dim - self.out_dim] = self.classifier.weight.data
            classifier.bias.data[:nb_output] = self.classifier.bias.data
            del self.classifier
        self.classifier = classifier

        if len(self.ts_nets) > 1:
            self.aux_classifier = SimpleLinear(new_dim, num_new_classes + 1, device=device)
        
            if self.predictor is None:
                self.predictor = SimpleLinear(self.ta_feature_dim, self.ta_feature_dim, device=device)
            
            if self.ts_attn is None:
                attn_num_heads = 8
                self.ts_attn = TSAttention(self.ta_feature_dim, attn_num_heads, device=device)
            else:
                self.ts_attn._reset_parameters()
                
            self.trans_classifier = SimpleLinear(self.ta_feature_dim, num_new_classes, device=device)

    def get_freezed_copy_ta(self):
        ta_net_copy = deepcopy(self.ta_net)
        for p in ta_net_copy.parameters():
            p.requires_grad_(False)
        return ta_net_copy.eval()

    def get_freezed_copy_projector(self):
        projector_copy = deepcopy(self.projector)
        for p in projector_copy.parameters():
            p.requires_grad_(False)
        return projector_copy.eval()

class TagFexLearner(Learner):
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
        self.memory_per_domain: int = args["memory_per_domain"]

        self.last_ta_net = None
        self.last_projector = None

        self.CL_type: str = CL_type

    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model.update_network(self.nb_classes, device=self.device)
        model = self.wrap_data_parallel(model)


        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

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
        best_model = model
        for epoch in range(self.base_epochs):
            print(
                f"Base Training - Epoch {epoch}/{self.base_epochs}",
                f"(Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']})",
            )
            model.train()
            for indecs, X, y,_ in tqdm(train_loader, "Training"):
                video, audio = X
                # video = video.repeat(2,1,1,1)
                # audio = audio.repeat(2,1)
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                video_label, audio_label, y = y
                # y = y.repeat(2)
                y: torch.Tensor = y.to(self.device, non_blocking=True)
                assert y.max() < baseset_size
                

                optimizer.zero_grad(set_to_none=True)
                outs = model(video, audio)
                logits, embedding = outs['logits'], outs['embedding']
                cls_loss = criterion(logits, y)
                infonce_loss = infoNCE_loss(embedding, 0.2)
                loss = cls_loss +  infonce_loss
                loss.backward()
                optimizer.step()
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
                best_model = model
                self.save_object(
                    model.state_dict(),
                    "model.pth"
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
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
            if best_acc > 0.99: break
        logging_file.close()
        model.train()
        self.model = best_model
        # self.model = self.load_object(model, "model.pth")

        self._construct_exemplar(self.model, train_loader, self.memory_per_domain)
        self.after_task()

    
    def learn(
        self,
        # data_loader: loader_t,
        dataset,
        incremental_size: int,
        phase: int,
        nb_classes: int,
        desc: str = "Incremental Learning",
    ) -> None:
        if desc == 'Re-align': return
        # if self.CL_type == 'CIL':
        self.nb_classes = nb_classes
        self.model.update_network(self.nb_classes, self.device)

        for i in range(phase):
            for p in self.model.ts_nets[i].parameters():
                p.requires_grad = False

        params = self.model.parameters()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        train_dataset = dataset.subset_at_phase(phase, self.memory) # 训练集中加入learner保存的样本
        data_loader = DataLoader(
            train_dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )

        self.model.train()
        for epoch in range(self.base_epochs):
            for indecs, X, y,_ in tqdm(data_loader, desc=desc):
                video, audio = X
                # video = video.repeat(2)
                # audio = audio.repeat(2)
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                video_label, audio_label, y = y
                # y = y.repeat(2)
                y: torch.Tensor = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(video, audio)
                logits, aux_logits, trans_logits, embedding, predicted_feature = outputs['logits'], outputs['aux_logits'], outputs['trans_logits'], outputs['embedding'], outputs['predicted_feature']
                cls_loss = criterion(logits, y)
                infonce_loss = infoNCE_loss(embedding, 0.2)
                aux_targets = y.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0
                )
                loss_aux = criterion(aux_logits, aux_targets)
                # old_ta_feature = self.last_ta_net(video.contiguous(), audio.contiguous())["features"]
                old_ta_feature = self.last_ta_net(video.contiguous(), audio.contiguous())["fmaps"].mean(1)
                kd_loss = infoNCE_distill_loss(self.last_projector(predicted_feature), self.last_projector(old_ta_feature), 0.2)
                cur_task_mask = (y >= self._known_classes)
                trans_cls_loss = criterion(trans_logits[cur_task_mask], y[cur_task_mask] - self._known_classes)
                
                if trans_cls_loss < cls_loss:
                    T = 2
                    transfer_loss = torch.nn.functional.kl_div((logits[cur_task_mask][:, self._known_classes:] / T).log_softmax(dim=1), (trans_logits.detach()[cur_task_mask] / T).softmax(dim=1), reduction='batchmean')
                else:
                    transfer_loss = torch.tensor(0., device=self.device)

                auto_kd_factor = self._known_classes / self.nb_classes
                loss = cls_loss + \
                2 * loss_aux + \
                (infonce_loss * (1 - auto_kd_factor) + 2 * kd_loss * auto_kd_factor) + \
                trans_cls_loss + \
                transfer_loss

                loss.backward()
                optimizer.step()
            scheduler.step()
        
        train_dataset = dataset.subset_at_phase(phase)
        data_loader = DataLoader(
            train_dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        self._construct_exemplar(self.model, data_loader, self.memory_per_domain)
        self.after_task()

    def after_task(self):
        self._known_classes = self.nb_classes
        self.last_ta_net = self.model.get_freezed_copy_ta()
        self.last_projector = self.model.get_freezed_copy_projector()

    def before_validation(self, phase) -> None:
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
        self.model = model

        self._construct_exemplar(data_loader, self.memory_per_domain)
        self._known_classes = self.nb_classes
        print('save old model')
        self.last_ta_net = self.model.get_freezed_copy_ta()
        self.last_projector = self.model.get_freezed_copy_projector()
    
    def update_model(self, phase=0, nb_classes=2, device="cpu"):
        if phase > 0:
            self.nb_classes = nb_classes
            self.model.update_network(self.nb_classes, self.device)
        else:
            self.nb_classes = nb_classes
            model = FusionModel(self.backbone, self.backbone_output, nb_classes).to(self.device, non_blocking=True)
            model = self.wrap_data_parallel(model)
            model.update_network(self.nb_classes, device=self.device)
            self.model = model
            self.model.eval()

class TSAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm_ts = nn.LayerNorm(embed_dim, device=device)
        self.norm_ta = nn.LayerNorm(embed_dim, device=device)

        self.weight_q = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_k_ts = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_k_ta = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_v_ts = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_v_ta = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
    
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.weight_q)
        nn.init.xavier_normal_(self.weight_k_ts)
        nn.init.xavier_normal_(self.weight_k_ta)
        nn.init.xavier_normal_(self.weight_v_ts)
        nn.init.xavier_normal_(self.weight_v_ta)

        self.norm_ta.reset_parameters()
        self.norm_ts.reset_parameters()
    
    def forward(self, ta_feats, ts_feats):
        bs, N, C = ta_feats.shape
        # feats: (bs, N, C)
        ta_feats = self.norm_ta(ta_feats)
        ts_feats = self.norm_ts(ts_feats)

        q = (ts_feats @ self.weight_q).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2) # (bs, H, N, Ch)
        k_ts = (ts_feats @ self.weight_k_ts).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k_ta = (ta_feats @ self.weight_k_ta).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ts = (ts_feats @ self.weight_v_ts).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ta = (ta_feats @ self.weight_v_ta).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        feat = nn.functional.scaled_dot_product_attention(q, torch.cat((k_ta, k_ts), dim=2), torch.cat((v_ta, v_ts), dim=2)) # (bs, H, N, Ch) # use default scale

        feat = feat.transpose(1, 2).flatten(2)

        return feat


class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

def infoNCE_loss(feats, t):
    cos_sim = torch.nn.functional.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll

def infoNCE_distill_loss(p_feats, z_feats, t):
    cos_sim = torch.nn.functional.cosine_similarity(p_feats[:,None,:], z_feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll
