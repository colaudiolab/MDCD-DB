
# -*- coding: utf-8 -*-

import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from torch.nn import DataParallel
from .Learner import Learner, loader_t
from sklearn.svm import LinearSVC

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

class FetrilLearner(Learner):
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        data_manager=None,
        CL_type: str='',
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        super().__init__(args, backbone, backbone_output,data_manager,  device, all_devices)
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
        self._known_domain = 0
        self._svm_accs = []
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
                    # loss1 = criterion(video_out, video_label)
                    # loss2 = criterion(audio_out, audio_label)
                    loss = criterion(logits, y)
                    # loss = loss1 + loss2 + loss3
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
        # self.model = model
        self.model = self.load_object(model, "model.pth")
        if hasattr(self.model, "module"):
            self.model = self.model.module

        self._compute_means(self.model, train_loader, 0)
        self._known_domain += 1
        self._build_feature_set()

        self._train_svm(self._feature_trainset)
        for testset in self._feature_testset:
            self._test_svm(testset)
    
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

        base_num_samples = dataset.original_dataset_length
        IL_batch_size = dataset.IL_batch_size
        num_workers = dataset.num_workers
        dataset = dataset.subset_at_phase(phase, self.memory)
        data_loader = DataLoader(
            dataset, batch_size=IL_batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
        self._compute_means(self.model, data_loader, phase)
        self._known_domain += 1
        self._build_feature_set()


        params = self.model.fc.parameters()
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
        #                    lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

        data_loader = DataLoader(self._feature_trainset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
        optimizer = torch.optim.SGD(params,momentum=0.9,lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max = self.base_epochs)
        criterion = torch.nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        self.model.backbone.eval()
        self.model.fc.train()
        for epoch in range(self.base_epochs):
            for indecs, X, y in tqdm(data_loader, desc=desc):
                X: torch.Tensor = X.to(self.device, non_blocking=True)
                y: torch.Tensor = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model.fc(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                break
            scheduler.step()

        # self._train_svm(self._feature_trainset)
        # for testset in self._feature_testset:
        #     self._test_svm(testset)


    def before_validation(self, phase: int) -> None:
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

    def _compute_means(self, model, loader, current_task):
        self.vectors_train = []
        self.labels_train = []
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (indecs, X, y,_) in enumerate(loader):
                video, audio = X
                video = video.to(self.device, non_blocking=True)
                audio = audio.to(self.device, non_blocking=True)
                # X: torch.Tensor = X.to(self.device, non_blocking=True)
                video_label, audio_label, target = y
                feature = model.feature(video, audio)
                # if feature.shape[0] == self.args.batch_size:
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
                break
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        self.vectors_train.append(features)
        self.labels_train.append(labels)
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
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)     
   
    def _build_feature_set(self):
        for domain_idx in range(0,self._known_domain):
            self.vectors_train.append(self.vectors_train[0] - self.prototype[self._known_domain] + self.prototype[domain_idx])
            self.labels_train.append(self.labels_train[0])
        
        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        self._feature_trainset = FeatureDataset(self.vectors_train,self.labels_train)

        # self.vectors_test = []
        # self.labels_test = []
        for domain_idx in range(0, self._known_domain):
            self._feature_testset = []
            test_dataset = self.data_manager.get_dataset(train=False).subset_at_phase(domain_idx)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=True)
            with torch.no_grad():
                labels = []
                features = []
                for i, (indecs, X, y,_) in enumerate(test_loader):
                    video, audio = X
                    video = video.to(self.device, non_blocking=True)
                    audio = audio.to(self.device, non_blocking=True)
                    video_label, audio_label, target = y
                    feature = self.model.feature(video, audio)
                    # if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
                    if i>10: break
                labels = np.array(labels)
                labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
                features = np.array(features)
                features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
                testset = FeatureDataset(features, labels)
                self._feature_testset.append(testset)
        #     self.vectors_test.append(features)
        #     self.labels_test.append(labels)
        # self.vectors_test = np.concatenate(self.vectors_test)
        # self.labels_test = np.concatenate(self.labels_test)

        # self._feature_testset = FeatureDataset(self.vectors_test,self.labels_test)
    
    def _train_svm(self,train_set):
        train_features = train_set.features.numpy()
        train_labels = train_set.labels.numpy()
        
        train_features = train_features/np.linalg.norm(train_features,axis=1)[:,None]
        self.svm_classifier = LinearSVC(random_state=42)
        self.svm_classifier.fit(train_features,train_labels)
        print("svm train: acc: {}".format(np.around(self.svm_classifier.score(train_features,train_labels)*100,decimals=2)))
        
    
    def _test_svm(self, test_set):
        test_features = test_set.features.numpy()
        test_labels = test_set.labels.numpy()

        test_features = test_features/np.linalg.norm(test_features,axis=1)[:,None]
        acc = self.svm_classifier.score(test_features,test_labels)
        self._svm_accs.append(np.around(acc*100,decimals=2))
        print("svm evaluation: acc_list: {}".format(self._svm_accs))
    
    def load_model(self, baseset_size, state_dict, data_loader):
        print('loading pretrained model')
        self.nb_classes = baseset_size
        model = FusionModel(self.backbone, self.backbone_output, baseset_size).to(self.device, non_blocking=True)
        model.load_state_dict(state_dict)
        model = self.wrap_data_parallel(model)
        self.model = model

        self._compute_means(self.model, data_loader, 0)
        self._known_domain += 1
        self._build_feature_set()

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



class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label