# -*- coding: utf-8 -*-
import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
from .Learner import Learner, loader_t
import numpy as np
from torch.utils.data import DataLoader
import copy

class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, nb_classes, CFL):
        super().__init__()
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.nb_classes = nb_classes
        self.fc = torch.nn.Linear(backbone_output, nb_classes)
        # self.video_fc = torch.nn.Linear(backbone_output, nb_classes)
        # self.audio_fc = torch.nn.Linear(backbone_output, nb_classes)
        self.CFL = CFL

    def forward(self, video, audio, train=True, phase=0):
        out = self.backbone(video, audio)
        video_out, audio_out, features, fmaps, video_fmaps, audio_fmaps = out['video'], out['audio'], out['features'], out['fmaps'], out['video_fmaps'], out['audio_fmaps']
        frequency = self.CFL.ecf(features)
        features = torch.angle(frequency)
        logits = self.fc(features)
        outputs = {'logits':logits, 'video':video_out, 'audio':audio_out, 'fmaps':fmaps, 'features':features, 'video_fmaps':video_fmaps, 'audio_fmaps':audio_fmaps}
        return outputs
        # video_out, audio_out, features, fmaps = out['video'], out['audio'], out['features'], out['fmaps']
        # x = self.fc(features)
        # outputs = {'logits':x, 'video':video_out, 'audio':audio_out, 'fmaps':fmaps, 'features':features}
        return outputs

    def feature(self, video, audio):
        out = self.backbone(video, audio)
        return out['features']

    def update_fc(self, nb_classes, device):
        fc = torch.nn.Linear(self.backbone_output, nb_classes)
        if self.fc is not None:
            nb_output = self.nb_classes
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc.to(device, non_blocking=True)
        self.nb_classes = nb_classes

class FrexLearner(Learner):
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
        self.memory = []
        self.memory_per_domain: int = args["memory_per_domain"] 
        if self.memory_per_domain == 0:
            self.memory_per_domain: float = args['memory_per_domain']
        self._known_domains = 0
        self.CL_type: str = CL_type
        self.nb_classes: int = 0
        self.domainFrequency = []
        self.CFL = ECFLoss()

    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size, self.CFL).to(self.device, non_blocking=True)
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
                    video_label, audio_label, y = y
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                    audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    
                    outs = model(video, audio)

                    video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
                    loss = criterion(logits, y)
                    loss.backward()

                    # video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
                    # loss: torch.Tensor = criterion(logits, y) + criterion(video_out, video_label) + criterion(audio_out, audio_label)
                    # loss.backward()
                    optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()

            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                self.save_object(
                    # (self.backbone, self.backbone_output),
                    # "backbone.pth",
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
                # train_meter.loss,
                # train_meter.accuracy,
                # train_meter.auc,
                # train_meter.f1_micro,
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
        logging_file.close()
        self.backbone.eval()
        # self.make_model()
        # self.model = model
        self.model = self.load_object(model, "model.pth")

        self._known_domains += 1    # 可能增加的域不止1个
        # self._reduce_exemplar(train_loader, self.memory_per_class)
        self._construct_exemplar(self.model, train_loader, self.memory_per_domain)

    
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
        self.update_model(phase=phase, nb_classes=nb_classes, device=self.device)

        params = self.model.parameters()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        train_dataset = dataset.subset_at_phase(phase, self.memory)
        data_loader = DataLoader(
            train_dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        self.model.train()
        for epoch in range(self.base_epochs):
            for indecs, X, y, _ in tqdm(data_loader, desc=desc, leave=False, ncols=50):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                video_label, audio_label, y = y
                y: torch.Tensor = y.to(self.device, non_blocking=True)
                video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outs = self.model(video, audio)
                # outs = self.model(video, audio)
                video_out, audio_out, logits, featureMap = outs['video'], outs['audio'], outs['logits'], outs['fmaps']
                
                # cls_loss = criterion(logits, y)
                # loss = cls_loss
                # loss.backward()

                cls_loss = criterion(logits, y)
                ecf_loss = self.domainFrequencyAligment(featureMap, y)
                loss = cls_loss + self.gamma * ecf_loss
                loss.backward()

                # video_out, audio_out, logits, featureMap = outs['video'], outs['audio'], outs['logits'], outs['fmaps']
                # cls_loss = criterion(logits, y) + criterion(video_out, video_label) + criterion(audio_out, audio_label)
                # ecf_loss = self.domainFrequencyAligment(featureMap, y)
                # # print(cls_loss, ecf_loss)
                # loss = cls_loss + self.lamb * ecf_loss
                # loss.backward()
                optimizer.step()
            scheduler.step()

        self._known_domains += 1
        if phase < 5:
            train_dataset = dataset.subset_at_phase(phase)
            data_loader = DataLoader(
                train_dataset, batch_size=2, shuffle=True, num_workers=num_workers, drop_last=True
            )
            self._construct_exemplar(self.model, data_loader, self.memory_per_domain)


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

    def before_training(self, dataset):
        pass

    def _reduce_exemplar(self, dataloader, m):
        pass
    
    def load_model(self, baseset_size, state_dict, data_loader):
        print('loading pretrained model')
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size, self.CFL).to(self.device, non_blocking=True)
        model.load_state_dict(state_dict)
        model = self.wrap_data_parallel(model)
        self.model = model

        self._construct_exemplar(self.model, data_loader, self.memory_per_domain)

    def update_model(self, phase=0, nb_classes=2, device="cpu"):
        if phase > 0:
            if self.CL_type == 'CIL':
                self.nb_classes = nb_classes
                self.model.update_fc(self.nb_classes, device)
        else:
            model = FusionModel(self.backbone, self.backbone_output, nb_classes).to(self.device, non_blocking=True)
            model = self.wrap_data_parallel(model)
            self.model = model
            self.model.eval()
    
    def _construct_exemplar(self, model, dataloader, m):
        model.eval()
        with torch.no_grad():
            class_exemplars = [[] for _ in range(self.nb_classes)]
            features = []
            labels = []
            frequencies = []
            for i, (indeces, X, y,index) in enumerate(dataloader):
                fn_img, fn_aud, label, start = index
                exemplar = list(zip(fn_img, fn_aud, label, start))
                _, _, target = y
                for j in range(len(exemplar)):
                    fn_img, fn_aud, label, start = exemplar[j]
                    label = label.item()
                    start = start.item()
                    idx = (fn_img, fn_aud, label, start)
                    class_exemplars[target[j]].append(idx)
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                outputs = model(video, audio)
                feature, featureMap = outputs['features'], outputs['fmaps']
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
                # extract the frequency of the embeddings
                frequency = self.CFL.ecf(featureMap)
                frequencies.append(frequency.cpu().numpy())
                # Clear buffer regularly
                if i % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        frequencies = np.array(frequencies)
        frequencies = np.reshape(frequencies, (frequencies.shape[0] * frequencies.shape[1], frequencies.shape[2]))
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        frequency_prototype = []
        features_prototype = []
        class_label = []
        if isinstance(m,float):
            total_num = sum([len(num) for num in class_exemplars])
            m = int(m*total_num)
            m = min(m, 500)
        if self.CL_type == 'CIL':
            labels_set = range(self._known_classes, self.nb_classes)
        total_selected_exemplars = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            data = class_exemplars[item]
            frequency_vectors = frequencies[index]
            frequency_mean = np.mean(frequency_vectors, axis=0)
            features_vectors = features[index]
            features_mean = np.mean(features_vectors, axis=0)
            frequency_prototype.append(frequency_mean)
            features_prototype.append(features_mean)


            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            length_of_vectors = features_vectors.shape[0]
            for k in range(1, min(m//2, length_of_vectors) + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (features_vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((features_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    data[i]
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(features_vectors[i])
                )  # New object to avoid passing by inference

                features_vectors = np.delete(
                    features_vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                # data = np.delete(
                #     data, i, axis=0
                # )  # Remove it to avoid duplicative selection
                del data[i]
            total_selected_exemplars.extend(selected_exemplars)
        self.memory.extend(total_selected_exemplars)
        # reserve the domain frequency prototype
        self.domainFrequency.append(frequency_prototype)
        print(f'Construct exemplars: {len(self.memory)}')
    
    def domainFrequencyAligment(self, featureMap, y):
        # extract the frequency of the embeddings
        frequency = self.CFL.ecf(featureMap)
        base_prototype = self.domainFrequency[0]

        labels = np.array(y.cpu())
        labels_set = np.unique(labels)
        ecf_loss = 0.0
        for label in labels_set:
            index = np.where(label == labels)[0]
            frequency = featureMap[index]
            label = 1 if label > 1 else label
            base_domain_prototype = torch.tensor(base_prototype[label], device=self.device)
            ecf_loss += self.CFL(frequency, base_domain_prototype)
        ecf_loss = ecf_loss / len(labels_set)
        return ecf_loss


class ECFLoss(torch.nn.Module):
    """
    ECF-Loss: 经验特征函数损失
    输入：s, T 均为 (B, N, d) 实数向量
    omega: 随机采样的一组频率向量 (N_omega, d)
    """
    def __init__(self, N_omega=256, d=256):
        super().__init__()
        # 随机初始化频率向量 (可训练或固定)
        self.omega = torch.nn.Parameter(torch.randn(N_omega, d) * 0.1, requires_grad=False)
        # self.omega = torch.nn.Parameter(torch.arange(1.0, N_omega+1, 1.0, dtype=torch.float32, requires_grad=False).unsqueeze(0))

    def ecf(self, x):
        """
        x: (B, N, d)
        return: (B, N_omega) 复数特征函数值
        """
        # x 形状 (B, N, d), omega 形状 (N_omega, d)
        # 内积 <omega, x> → (B, N, N_omega)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        inner = torch.einsum('bnd,od->bno', x, self.omega)
        # 平均特征函数 (B, N_omega) 复数
        ecf_val = torch.mean(torch.exp(1j * inner), dim=1)
        return ecf_val
    
    # def ecf(self, x):
    #     """
    #     x: (B, N)
    #     return: (B, N_omega) 复数特征函数值
    #     """
    #     # x 形状 (B, N, d), omega 形状 (N_omega, d)
    #     # 内积 <omega, x> → (B, N, N_omega)
    #     if len(x.shape) == 2:
    #         x = x.unsqueeze(-1)
    #     inner = x @ self.omega
    #     # 平均特征函数 (B, N_omega) 复数
    #     ecf_val = torch.mean(torch.exp(1j * inner), dim=1)
    #     return ecf_val

    def forward(self, s, phi_T):
        """
        s, T: (B, N, d) 实数向量
        T: 已经是频域特征
        return: scalar loss
        """
        phi_s = self.ecf(s)
        # phi_T = self.ecf(T)
        # 复数差的模方 |phi_s - phi_T|^2
        loss = torch.mean(torch.abs(phi_s - phi_T) ** 2)
        return loss

def extract_spatial_ecf(video):
    """空间ECF提取（向量化）"""
    batch_size, length_channels, height, width = video.shape
    
    # 重塑为 [batch*length*channels, height*width]
    video_flat = video.reshape(batch_size * length_channels, height * width)
    
    # 计算ECF相位
    # N_omega = height * width
    # d = 128*128
    # omega = torch.nn.Parameter(torch.randn(128*128, d) * 0.1, requires_grad=False)
    t_values = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0, 5.0], device=video.device)
    n_t = len(t_values)
    phase_features_flat = torch.zeros(batch_size * length_channels, 
                                        n_t, height * width, device=video.device)
    
    
    for k, t_val in enumerate(t_values):
        # 向量化计算
        exp_values = torch.exp(1j * t_val * video_flat)  # [N, H*W]
        
        # 考虑局部邻域（3x3窗口）
        # 将扁平化的图像重塑回2D
        exp_values_2d = exp_values.view(batch_size * length_channels, height, width)
        
        # 应用平均池化（模拟局部ECF）
        kernel_size = 3
        padding = kernel_size // 2
        
        # 展开为滑动窗口
        unfolded = torch.nn.functional.unfold(
            exp_values_2d.unsqueeze(1).float(),  # 添加通道维度
            kernel_size=kernel_size,
            padding=padding
        )  # [N, kernel_size*kernel_size, H*W]
        
        # 计算局部平均（ECF）
        local_ecf = torch.mean(unfolded, dim=1)  # [N, H*W]
        
        # 计算相位
        phase = torch.angle(local_ecf.to(torch.complex64))
        phase_features_flat[:, k, :] = phase
    phase_features_flat = phase_features_flat.mean(1)

    # 重塑回原始形状
    phase_features = phase_features_flat.view(
        batch_size, -1, 3, height, width
    )# [batch, length, n_t, channels, height, width]

    # visual_phase(video, phase_features, 0, 0, 0)
    
    return phase_features_flat.view(batch_size, length_channels, height, width)

def visual_phase(video, ecf_phase_frame, batch_idx, frame_idx, channel_idx):
    import numpy as np
    import matplotlib.pyplot as plt

    batch, length_channels, height, width = video.shape
    original_frame = video.view(batch, -1, 3, height, width)[batch_idx, frame_idx].permute(1, 2, 0).cpu().numpy()
    phase_frame = ecf_phase_frame[batch_idx, frame_idx].permute(1, 2, 0).cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8,4))

    ax1 = axes[0]
    im1 = ax1.imshow(original_frame)
    ax1.set_title(f'原始帧 (batch={batch_idx}, frame={frame_idx})')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. ECF相位图
    ax2 = axes[1]
    im2 = ax2.imshow(phase_frame, vmin=-np.pi, vmax=np.pi)
    ax2.set_title(f'ECF相位 (t={0})')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    import random
    n = random.randint(0,10)
    plt.savefig(f'show_{n}.png')
