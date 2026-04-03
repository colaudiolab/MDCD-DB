# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import StepLR
# from torch.autograd import Variable
# import os
# import sys
# import numpy as np
# from myNetwork import network
# from iCIFAR100 import iCIFAR100


# class protoAugSSL:
#     def __init__(self, args, file_name, feature_extractor, task_size, device):
#         self.file_name = file_name
#         self.args = args
#         self.epochs = args.epochs
#         self.learning_rate = args.learning_rate
#         self.model = network(args.fg_nc*4, feature_extractor)
#         self.radius = 0
#         self.prototype = None
#         self.class_label = None
#         self.numclass = args.fg_nc
#         self.task_size = task_size
#         self.device = device
#         self.old_model = None
    
#     def beforeTrain(self, current_task):
#         self.model.eval()
#         if current_task == 0:
#             classes = [0, self.numclass]
#         else:
#             classes = [self.numclass-self.task_size, self.numclass]
#         self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
#         if current_task > 0:
#             self.model.Incremental_learning(4*self.numclass)
#         self.model.train()
#         self.model.to(self.device)

#     def _get_test_dataloader(self, classes):
#         self.test_dataset.getTestData_up2now(classes)
#         test_loader = DataLoader(dataset=self.test_dataset,
#                                  shuffle=True,
#                                  batch_size=self.args.batch_size)
#         return test_loader

#     def train(self, current_task, old_class=0):
#         opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
#         scheduler = StepLR(opt, step_size=45, gamma=0.1)
#         accuracy = 0
#         for epoch in range(self.epochs):
#             scheduler.step()
#             for step, (indexs, images, target) in enumerate(self.train_loader):
#                 images, target = images.to(self.device), target.to(self.device)

#                 # self-supervised learning based label augmentation
#                 images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
#                 images = images.view(-1, 3, 32, 32)
#                 target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

#                 opt.zero_grad()
#                 loss = self._compute_loss(images, target, old_class)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()
#             if epoch % self.args.print_freq == 0:
#                 accuracy = self._test(self.test_loader)
#                 print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
#         self.protoSave(self.model, self.train_loader, current_task)

#     def _test(self, testloader):
#         self.model.eval()
#         correct, total = 0.0, 0.0
#         for setp, (indexs, imgs, labels) in enumerate(testloader):
#             imgs, labels = imgs.to(self.device), labels.to(self.device)
#             with torch.no_grad():
#                 outputs = self.model(imgs)
#             outputs = outputs[:, ::4]  # only compute predictions on original class nodes
#             predicts = torch.max(outputs, dim=1)[1]
#             correct += (predicts.cpu() == labels.cpu()).sum()
#             total += len(labels)
#         accuracy = correct.item() / total
#         self.model.train()
#         return accuracy

#     def _compute_loss(self, imgs, target, old_class=0):
#         output = self.model(imgs)
#         output, target = output.to(self.device), target.to(self.device)
#         loss_cls = nn.CrossEntropyLoss()(output/self.args.temp, target)
#         if self.old_model is None:
#             return loss_cls
#         else:
#             feature = self.model.feature(imgs)
#             feature_old = self.old_model.feature(imgs)
#             loss_kd = torch.dist(feature, feature_old, 2)

#             proto_aug = []
#             proto_aug_label = []
#             index = list(range(old_class))
#             for _ in range(self.args.batch_size):
#                 np.random.shuffle(index)
#                 temp = self.prototype[index[0]] + np.random.normal(0, 1, 512) * self.radius
#                 proto_aug.append(temp)
#                 proto_aug_label.append(4*self.class_label[index[0]])

#             proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
#             proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
#             soft_feat_aug = self.model.fc(proto_aug)
#             loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug/self.args.temp, proto_aug_label)

#             return loss_cls + self.args.protoAug_weight*loss_protoAug + self.args.kd_weight*loss_kd

#     def afterTrain(self):
#         path = self.args.save_path + self.file_name + '/'
#         if not os.path.isdir(path):
#             os.makedirs(path)
#         self.numclass += self.task_size
#         filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
#         torch.save(self.model, filename)
#         self.old_model = torch.load(filename)
#         self.old_model.to(self.device)
#         self.old_model.eval()

#     def protoSave(self, model, loader, current_task):
#         features = []
#         labels = []
#         model.eval()
#         with torch.no_grad():
#             for i, (indexs, images, target) in enumerate(loader):
#                 feature = model.feature(images.to(self.device))
#                 if feature.shape[0] == self.args.batch_size:
#                     labels.append(target.numpy())
#                     features.append(feature.cpu().numpy())
#         labels_set = np.unique(labels)
#         labels = np.array(labels)
#         labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
#         features = np.array(features)
#         features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
#         feature_dim = features.shape[1]

#         prototype = []
#         radius = []
#         class_label = []
#         for item in labels_set:
#             index = np.where(item == labels)[0]
#             class_label.append(item)
#             feature_classwise = features[index]
#             prototype.append(np.mean(feature_classwise, axis=0))
#             if current_task == 0:
#                 cov = np.cov(feature_classwise.T)
#                 radius.append(np.trace(cov) / feature_dim)

#         if current_task == 0:
#             self.radius = np.sqrt(np.mean(radius))
#             self.prototype = prototype
#             self.class_label = class_label
#             print(self.radius)
#         else:
#             self.prototype = np.concatenate((prototype, self.prototype), axis=0)
#             self.class_label = np.concatenate((class_label, self.class_label), axis=0)

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
import torch.nn as nn
import copy
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

class PassLearner(Learner):
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
        CL_type: str='',
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        super().__init__(args, backbone, backbone_output, data_manager, device, all_devices)
        self.learning_rate: float = args["learning_rate"]
        self.buffer_size: int = args["buffer_size"]
        self.gamma: float = args["gamma"]
        self.base_epochs: int = args["base_epochs"]
        self.warmup_epochs: int = args["warmup_epochs"]
        self.temp = 0.1
        self.protoAug_weight = 10.0
        self.kd_weight = 10.0
        self.old_model = None
        self.batch_size = args["batch_size"]
        self.radius = 0
        self.prototype = None
        self.class_label = None
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
        self.model = model


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
                for indeces,X, y,_ in tqdm(train_loader, "Training"):
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
                    # logits = model(X)
                    # video_out, audio_out, logits = model(video, audio)
                    # loss1 = criterion(video_out, video_label)
                    # loss2 = criterion(audio_out, audio_label)
                    # loss3 = criterion(logits, y)
                    # loss = loss1 + loss2 + loss3
                    loss = self._compute_loss(video, audio, y)
                    # loss: torch.Tensor = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    break
                scheduler.step()

            # Validation on training set
            model.eval()
            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                self.save_object(
                    # (self.backbone, X.shape[1], self.backbone_output),
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
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
        logging_file.close()
        # self.backbone.eval()
        # self.make_model()
        # self.model = model

        self.model = self.load_object(model, "model.pth")
        self.protoSave(self.model, train_loader, current_task=0)
        self.afterTrain(phase=0)
    
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

        base_num_samples = dataset.original_dataset_length
        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        dataset = dataset.subset_at_phase(phase)

        data_loader = DataLoader(
            dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        self.model.train()
        for epoch in range(self.base_epochs):
            for indecs, X, y,_ in tqdm(data_loader, desc=desc):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                # X: torch.Tensor = X.to(self.device, non_blocking=True)
                video_label, audio_label, y = y
                y: torch.Tensor = y.to(self.device, non_blocking=True)
                video_label: torch.Tensor = video_label.to(self.device, non_blocking=True)
                audio_label: torch.Tensor = audio_label.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                # self.model.fit(video, audio, y, increase_size=incremental_size)
                # video_out, audio_out, logits = self.model(video, audio)
                # loss1 = criterion(video_out, video_label)
                # loss2 = criterion(audio_out, audio_label)
                # loss3 = criterion(logits, y)
                # loss = loss1 + loss2 + loss3
                loss = self._compute_loss(video, audio, y, phase*2)
                loss.backward()
                optimizer.step()
                break
            scheduler.step()

        self.protoSave(self.model, data_loader, current_task=phase)
        self.afterTrain(phase=phase)

    def before_validation(self, phase) -> None:
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

    def _compute_loss(self, video, audio, target, old_class=0):
        outs = self.model(video, audio)
        video_out, audio_out, output = outs['video'], outs['audio'], outs['logits']
        # _, _, output = self.model(video, audio)
        output, target = output.to(self.device), target.to(self.device)
        loss_cls = nn.CrossEntropyLoss()(output, target)
        if self.old_model is None:
            return loss_cls
        else:
            feature = self.model.feature(video, audio)
            feature_old = self.old_model.feature(video, audio)
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            for _ in range(self.batch_size):
                np.random.shuffle(index)
                temp = self.prototype[index[0]] + np.random.normal(0, 1, 256) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(self.class_label[index[0]])
            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.fc(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug/self.temp, proto_aug_label)

            return loss_cls + self.protoAug_weight*loss_protoAug + self.kd_weight*loss_kd

    def afterTrain(self, phase):
        print('save old model')
        self.old_model = copy.deepcopy(self.model)
        self.old_model.to(self.device)
        self.old_model.eval()

        if isinstance(self.model, DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        self.save_object(
            model.state_dict(),
            f"model_{phase}.pth"
        )

    def protoSave(self, model, loader, current_task):
        print('save prototype')
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (indeces, X, y,_) in enumerate(loader):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                # X: torch.Tensor = X.to(self.device, non_blocking=True)
                video_label, audio_label, target = y
                feature = model.feature(video, audio)
                # if feature.shape[0] == self.args.batch_size:
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
                if i > 10 : break
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
    
    def load_model(self, baseset_size, state_dict, data_loader):
        print('loading pretrained model')
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model.load_state_dict(state_dict)
        model = self.wrap_data_parallel(model)
        self.model = model

        self.protoSave(self.model, data_loader, current_task=0)

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