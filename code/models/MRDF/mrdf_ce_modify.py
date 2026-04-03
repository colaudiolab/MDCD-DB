from typing import Dict, List, Optional, Union, Sequence

import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Module, TransformerEncoder, TransformerEncoderLayer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchmetrics
from .resnet import ResEncoder

def Average(lst):
    return sum(lst) / len(lst)


def Opposite(a):
    a = a + 1
    a[a>1.5] = 0
    return a


class ContrastLoss(Module):

    def __init__(self, loss_fn: Module, margin: float = 0.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = loss_fn

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor):
        # input: (B, C, T)
        loss = []
        for i in range(pred1.shape[0]):
            # mean L2 distance squared
            d = self.loss_fn(pred1[i, :], pred2[i, :])
            # d = self.cosim(pred1[i, :], pred2[i, :])
            if labels[i]:
                # if is positive pair, minimize distance
                loss.append(1 - d)
            else:
                # if is negative pair, minimize (margin - distance) if distance < margin
                loss.append(torch.clip((d - self.margin), min=0.))
        return torch.mean(torch.stack(loss))

class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, 768)
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1, activation='gelu'), num_layers=12)

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        if self.encoder is not None:
            x = self.encoder(x).transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x

class MRDF_CE(Module):

    def __init__(self,
       margin_contrast=0.0, weight_decay=0.0001, learning_rate=0.0002, distributed=False):
        super().__init__()

        self.embed = 768
        encoder_embed_dim = 768
        self.dropout = 0.1
        dropout_input = 0.0

        resnet = ResEncoder(relu_type='prelu', weights=None)
        self.feature_extractor_audio_hubert = SubModel(resnet=None, input_dim=80)
        self.feature_extractor_video_hubert = SubModel(resnet=resnet, input_dim=resnet.backend_out)

        self.project_audio = nn.Sequential(torch.nn.LayerNorm(self.embed), nn.Linear(self.embed, self.embed),
                                           nn.Dropout(self.dropout))

        self.project_video = nn.Sequential(torch.nn.LayerNorm(self.embed), nn.Linear(self.embed, self.embed),
                                           nn.Dropout(self.dropout))

        post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim
            else nn.Identity() 
        )
        self.project_hubert = nn.Sequential(nn.LayerNorm(self.embed), post_extract_proj,
                                           nn.Dropout(dropout_input))

        self.fusion_encoder_hubert = TransformerEncoder(TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1, activation='gelu'), num_layers=12)

        self.final_proj_audio = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.final_proj_video = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.final_proj_hubert = nn.Linear(encoder_embed_dim, encoder_embed_dim)

        self.video_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        self.audio_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        # #
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed))
        self.mm_classifier = nn.Sequential(nn.Linear(self.embed, self.embed), nn.ReLU(inplace=True),
                                              nn.Linear(self.embed, 2))

        self.contrast_loss = ContrastLoss(loss_fn=nn.CosineSimilarity(dim=-1), margin=margin_contrast)
        self.mm_cls = CrossEntropyLoss()
        self.a_cls = CrossEntropyLoss()
        self.v_cls = CrossEntropyLoss()

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed

        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.auroc = torchmetrics.classification.BinaryAUROC(thresholds=None)
        self.f1score = torchmetrics.classification.BinaryF1Score()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.precisions = torchmetrics.classification.BinaryPrecision()

        self.best_loss = 1e9
        self.best_acc, self.best_auroc = 0.0, 0.0
        self.best_real_f1score, self.best_real_recall,  self.best_real_precision = 0.0, 0.0, 0.0
        self.best_fake_f1score, self.best_fake_recall, self.best_fake_precision = 0.0, 0.0, 0.0

        self.softmax = nn.Softmax(dim=1)

    def forward(self, video, audio):
        # video, audio = inp
        a_features = self.feature_extractor_audio_hubert(audio).transpose(1, 2)
        v_features = self.feature_extractor_video_hubert(video).transpose(1, 2)
        
        av_features = torch.cat([a_features, v_features], dim=1)

        a_cross_embeds = a_features.mean(1)
        v_cross_embeds = v_features.mean(1)

        a_features = self.project_audio(a_features)
        v_features = self.project_video(v_features)
        av_features = self.project_hubert(av_features)

        a_embeds = a_features.mean(1)
        v_embeds = v_features.mean(1)

        a_embeds = self.audio_classifier(a_embeds)
        v_embeds = self.video_classifier(v_embeds)

        cls_token = self.cls_token.expand(av_features.shape[0], -1, -1)
        av_features = torch.cat([cls_token, av_features], dim=1)
        av_features = self.fusion_encoder_hubert(av_features)
        return av_features[:, 0, :], v_cross_embeds, a_cross_embeds, v_embeds, a_embeds
        # m_logits = self.mm_classifier(av_features[:, 0, :])

        # return m_logits, v_cross_embeds, a_cross_embeds, v_embeds, a_embeds

    def get_avg_feat(self, feat, mask):
        mask_un = mask.to(dtype=torch.float).unsqueeze(1)
        feat = feat * mask_un
        mask_un_sum = torch.sum(mask_un, dim=1, dtype=torch.float)
        mask_un_sum[mask_un_sum == 0.] = 1.
        feat = torch.sum(feat, dim=1) / mask_un_sum
        return feat

    def loss_fn(self, m_logits, v_feats, a_feats, v_logits, a_logits, v_label, a_label, c_label, m_label) -> Dict[str, Tensor]:

        contrast_loss = self.contrast_loss(v_feats, a_feats, c_label)
        a_loss = self.a_cls(a_logits, a_label)
        v_loss = self.v_cls(v_logits, v_label)

        mm_loss = self.mm_cls(m_logits, m_label)
        loss = mm_loss + a_loss + v_loss + contrast_loss

        return {"loss": loss, "mm_loss": mm_loss}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:

        m_logits, v_feats, a_feats, v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_feats, a_feats, v_logits, a_logits, batch['v_label'], batch['a_label'],
                                               batch['c_label'], batch['m_label'])
        
        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)

        return {"loss": loss_dict["loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:

        m_logits, v_feats, a_feats, v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_feats, a_feats, v_logits, a_logits, batch['v_label'], batch['a_label'],
                                               batch['c_label'], batch['m_label'])
        
        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)

        return {"loss": loss_dict["mm_loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def test_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
                        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
                        ) -> Tensor:

        m_logits, v_feats, a_feats, v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_feats, a_feats, v_logits, a_logits, batch['v_label'], batch['a_label'],
                                               batch['c_label'], batch['m_label'])
        
        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        return {"loss": loss_dict["mm_loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}


    def training_step_end(self, training_step_outputs):
        # others: common, ensemble, multi-label
        train_acc = self.acc(training_step_outputs['preds'], training_step_outputs['targets']).item()
        train_auroc = self.auroc(training_step_outputs['preds'], training_step_outputs['targets']).item()
        
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_auroc", train_auroc, prog_bar=True)

    def validation_step_end(self, validation_step_outputs):
        # others: common, ensemble, multi-label
        val_acc = self.acc(validation_step_outputs['preds'], validation_step_outputs['targets']).item()
        val_auroc = self.auroc(validation_step_outputs['preds'], validation_step_outputs['targets']).item()
        
        self.log("val_re", val_acc+val_auroc, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_auroc", val_auroc, prog_bar=True)

    def training_epoch_end(self, training_step_outputs):
        train_loss = Average([i["loss"] for i in training_step_outputs]).item()
        preds = [item for list in training_step_outputs for item in list["preds"]]
        targets = [item for list in training_step_outputs for item in list["targets"]]
        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        train_acc = self.acc(preds, targets).item()
        train_auroc = self.auroc(preds, targets).item()

        print("Train - loss:", train_loss, "acc: ", train_acc, "auroc: ", train_auroc)


    def validation_epoch_end(self, validation_step_outputs):
        valid_loss = Average([i["loss"] for i in validation_step_outputs]).item()
        preds = [item for list in validation_step_outputs for item in list["preds"]]
        targets = [item for list in validation_step_outputs for item in list["targets"]]
        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        self.best_acc = self.acc(preds, targets).item()
        self.best_auroc = self.auroc(preds, targets).item()
        self.best_real_f1score = self.f1score(preds, targets).item()
        self.best_real_recall = self.recall(preds, targets).item()
        self.best_real_precision = self.precisions(preds, targets).item()

        self.best_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        self.best_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        self.best_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        self.best_loss = valid_loss
        print("Valid loss: ", self.best_loss, "acc: ", self.best_acc, "auroc: ", self.best_auroc,
              "real_f1score:",
              self.best_real_f1score, "real_recall: ", self.best_real_recall, "real_precision: ",
              self.best_real_precision, "fake_f1score: ", self.best_fake_f1score, "fake_recall: ",
              self.best_fake_recall, "fake_precision: ", self.best_fake_precision)

    def test_epoch_end(self, test_step_outputs):
        test_loss = Average([i["loss"] for i in test_step_outputs]).item()
        preds = [item for list in test_step_outputs for item in list["preds"]]
        targets = [item for list in test_step_outputs for item in list["targets"]]

        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        test_acc = self.acc(preds, targets).item()
        test_auroc = self.auroc(preds, targets).item()
        test_real_f1score = self.f1score(preds, targets).item()
        test_real_recall = self.recall(preds, targets).item()
        test_real_precision = self.precisions(preds, targets).item()

        test_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        test_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        test_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        self.log("test_acc", test_acc)
        self.log("test_auroc", test_auroc)
        self.log("test_real_f1score", test_real_f1score)
        self.log("test_real_recall", test_real_recall)
        self.log("test_real_precision", test_real_precision)
        self.log("test_fake_f1score", test_fake_f1score)
        self.log("test_fake_recall", test_fake_recall)
        self.log("test_fake_precision", test_fake_precision)
        return {"loss": test_loss, "test_acc": test_acc, "auroc": test_auroc, "real_f1score": test_real_f1score,
                "real_recall": test_real_recall, "real_precision":
                    test_real_precision, "fake_f1score": test_fake_f1score, "fake_recall": test_fake_recall,
                "fake_precision": test_fake_precision}


    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }
