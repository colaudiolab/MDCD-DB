# -*- coding: utf-8 -*-
"""
Our Implementation of the DFIL (https://github.com/DeepFakeIL/DFIL)
"""

import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import numpy as np
import math
import copy
from .Learner import Learner, loader_t
torch.multiprocessing.set_sharing_strategy('file_system')

class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, nb_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.nb_classes = nb_classes
        self.fc = torch.nn.Linear(backbone_output, nb_classes)

    def forward(self, video, audio, train=True, phase=0):
        out = self.backbone(video, audio)
        video_out, audio_out, fusion_feature = out['video'], out['audio'], out['features']
        x = self.fc(fusion_feature)
        outputs = {'logits':x, 'video':video_out, 'audio':audio_out, 'features':fusion_feature}
        return outputs
    def feature(self, video, audio):
        video_out, audio_out, x = self.backbone(video, audio)
        return x
    
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

class DFILLearner(Learner):
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
        self.memory_per_domain: int = args["memory_per_domain"]   # 100 for class incremental learning; x for domain incremental learning
        self.memory = []
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

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                           lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)
        criterion_supcon = SupConLoss(self.device)

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

        for epoch in range(self.base_epochs):
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
                    video_out, audio_out, logits, fusion_feature = outs['video'], outs['audio'], outs['logits'], outs['features']
                    # video_out, audio_out, fusion_feature, logits = model(video, audio, ret_feature=True)
                    # fusion_feature = torch.nn.functional.adaptive_avg_pool2d(fusion_feature, (1, 1)).view(fusion_feature.size(0), -1)
                    loss1 = criterion(logits, y)
                    loss2 = criterion_supcon(fusion_feature, y)
                    loss = loss1 + loss2
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()
            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
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
        # self.model = model
        self.model = self.load_object(model, "model.pth")
        self._construct_exemplar(train_loader, self.memory_per_domain)
        self.afterTrain(phase=0)
    
    def learn(
        self,
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
        dataset = dataset.subset_at_phase(phase, self.memory)
        data_loader = DataLoader(
            dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )

        self.criterion_supcon = SupConLoss(self.device)

        self.model.train()
        for epoch in range(self.base_epochs):
            for indecs, X, y,_ in tqdm(data_loader, desc=desc):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                video_label, audio_label, y = y
                y: torch.Tensor = y.to(self.device, non_blocking=True)
                video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outs = self.model(video, audio)
                video_out, audio_out, logits, fusion_feature = outs['video'], outs['audio'], outs['logits'], outs['features']
                tea_outs = self.teacher_model(video, audio)
                video_out, audio_out, teacher_logits, teacher_fusion_feature = tea_outs['video'], tea_outs['audio'], tea_outs['logits'], tea_outs['features']
                # video_out, audio_out, fusion_feature, logits = self.model(video, audio, ret_feature=True)
                # video_out, audio_out, teacher_fusion_feature, teacher_logits = self.teacher_model(video, audio, ret_feature=True)

                loss_ce = criterion(logits, y)
                loss_fd = 0.01 * self.loss_FD(fusion_feature,teacher_fusion_feature)
                loss_kd = self.loss_fn_kd(logits, y, teacher_logits, T=20.0, alpha=0.3)
                loss_consup = 0.1 * self.loss_ConSup(fusion_feature,y)

                loss = loss_ce + loss_fd + loss_kd + loss_consup 
                loss.backward()
                optimizer.step()
            scheduler.step()
        
        
        self._construct_exemplar(data_loader, self.memory_per_domain)
        self.afterTrain(phase=phase)

    def before_validation(self, phase: int) -> None:
        pass

    def inference(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.model(video, audio)

    @torch.no_grad()
    def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.all_devices is not None and len(self.all_devices) > 1:
            return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
        return model

    def afterTrain(self, phase):
        self._known_classes = self.nb_classes
        print('save old model')
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        if isinstance(self.model, DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        self.save_object(
            model.state_dict(),
            f"model_{phase}.pth"
        )

    def _construct_exemplar(self, dataloader, m):
        self.model.eval()
        # calculate feature mean
        with torch.no_grad():
            class_exemplars = [[]]* self.nb_classes
            features = []
            labels = []
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
                feature = self.model(video, audio)['features']
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = []
        class_label = []
        
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            vectors = features[index]
            class_mean = np.mean(vectors, axis=0)
            prototype.append(class_mean)
        prototype = torch.tensor(prototype).to(self.device)
        # calculate cross-entropy and distance
        with torch.no_grad():
            distances = []
            entropys = []
            for i, (indeces, X, y,index) in enumerate(dataloader):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                video_label, audio_label, label = y
                outs = self.model(video, audio)
                logits, feature = outs['logits'], outs['features']
                # _,_, feature, logits = self.model(video, audio, ret_feature=True)

                prob = torch.nn.functional.softmax(logits.data,dim=1)
                # feature = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1)).view(feature.size(0), -1)
                for i in range(feature.shape[0]):
                    # if logits[i][0].item() <= 0:
                    #     prob[i][0] = 1e-40
                    # if logits[i][1].item() <= 0:
                    #     prob[i][1] = 1e-40
                    # entropys.append(- (prob[i][0].item() * math.log(prob[i][0].item()) + prob[i][1].item() * math.log(prob[i][1].item())) )
                    entropy = -sum(p.item() * math.log(p.item()) for p in prob[i] if p > 0) # multi class
                    entropys.append(entropy)
                    now_feature = feature[i].unsqueeze(0)
                    # if label[i] == 0:
                    #     distance = torch.nn.functional.pairwise_distance(prototype[0], now_feature, p=2)
                    # else:
                    #     distance = torch.nn.functional.pairwise_distance(prototype[1], now_feature, p=2)
                    distance = torch.nn.functional.pairwise_distance(prototype[label[i]], now_feature, p=2)
                    distances.append(distance)

        # Select
        # exemplars = np.array(exemplars)
        entropys = torch.tensor(entropys)
        distances = torch.tensor(distances)
        if self.CL_type == 'CIL':
            labels_set = range(self._known_classes, self.nb_classes)
        for item in labels_set:
            index = np.where(item == labels)[0]
            data = class_exemplars[item]
            entropys_label = entropys[index]
            distances_label = distances[index]

            top_entropys, top_entropys_indices = torch.topk(entropys_label, m//2, largest=True)
            top_distances, top_distances_indices = torch.topk(distances_label, m//2, largest=False)

            for idx in top_entropys_indices:
                self.memory.append(data[idx])
            for idx in top_distances_indices:
                self.memory.append(data[idx])

        print(f'Memory size: {len(self.memory)}')

    def loss_FD(self, Student_feature, Teacher_feature):
        # Student_feature = torch.nn.functional.adaptive_avg_pool2d(Student_feature, (1, 1)) 
        Student_feature = Student_feature.view(Student_feature.size(0), -1)
        Student_feature = torch.nn.functional.normalize(Student_feature, dim=1)

        # Teacher_feature = torch.nn.functional.adaptive_avg_pool2d(Teacher_feature, (1, 1)) 
        Teacher_feature = Teacher_feature.view(Teacher_feature.size(0), -1)
        Teacher_feature = torch.nn.functional.normalize(Teacher_feature, dim=1)

        loss = torch.nn.functional.mse_loss(Student_feature, Teacher_feature, reduction="mean")
        return loss

    def loss_fn_kd(self, y, labels, teacher_scores, T, alpha):
        return torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(y[:,:self.nb_classes-1]/T,dim=1), torch.nn.functional.log_softmax(teacher_scores/T,dim=1)) * (T*T * 2.0 * alpha) + torch.nn.functional.cross_entropy(y, labels) * (1. - alpha)

    def loss_ConSup(self, fc_features,labels):
        # fc_features = torch.nn.functional.adaptive_avg_pool2d(fc_features, (1, 1)) 
        fc_features = fc_features.view(fc_features.size(0), -1)
        loss = self.criterion_supcon(fc_features,labels)

        return loss

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
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

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

class SupConLoss(torch.nn.Module):

    def __init__(self, device, temperature=1, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
        输出:
            loss值
        """
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask 

        logits_mask = torch.ones_like(mask)- torch.eye(batch_size).to(self.device)  
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''        
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2] 
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss