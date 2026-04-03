# -*- coding: utf-8 -*-

import torch
from typing import Dict, Tuple, Union, Optional, Callable

# from torchvision.models import WeightsEnum
from torch.nn import Flatten

from torchvision.models.resnet import (
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    # ResNet18_Weights,
    # ResNet34_Weights,
    # ResNet50_Weights,
    # ResNet101_Weights,
    # ResNet152_Weights,
)

from .CifarResNet import (
    CifarResNet,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet1202,
)

# from MRDF.mrdf_ce import MRDF_CE
from .Fine_grained.graph_video_audio_model import GAT_video_audio
from .MMTrace import MMTrace

from torchvision.models.vision_transformer import (
    VisionTransformer,
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
    vit_h_14,
    # ViT_B_16_Weights,
    # ViT_B_32_Weights,
    # ViT_L_16_Weights,
    # ViT_L_32_Weights,
    # ViT_H_14_Weights,
)

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "CifarResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
    "VisionTransformer",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
    "load_backbone",
    # MRDF_CE,
    GAT_video_audio,
    MMTrace,
]

# fmt: off
models: Dict[str, Tuple[
        int,    # Input image size
        Callable[[], Union[CifarResNet, ResNet, VisionTransformer, Flatten]], # Model constructor
        # Optional[WeightsEnum]
        Optional[int]
    ]
] = {
    # # MNIST: No backbone
    # "Flatten": (28, Flatten, None),
    # # ResNet for CIFAR
    # "resnet20":   (32, resnet20  , None),
    # "resnet32":   (32, resnet32  , None),
    # "resnet44":   (32, resnet44  , None),
    # "resnet56":   (32, resnet56  , None),
    # "resnet110":  (32, resnet110 , None),
    # "resnet1202": (32, resnet1202, None),
    # # ResNet for ImageNet
    # "resnet18":  (224, resnet18,  ResNet18_Weights.DEFAULT ),
    # "resnet34":  (224, resnet34,  ResNet34_Weights.DEFAULT ),
    # "resnet50":  (224, resnet50,  ResNet50_Weights.DEFAULT ),
    # "resnet101": (224, resnet101, ResNet101_Weights.DEFAULT),
    # "resnet152": (224, resnet152, ResNet152_Weights.DEFAULT),
    # # Vision Transformer for ImageNet
    # "vit_b_16": (384, vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    # "vit_b_32": (224, vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1         ),
    # "vit_l_16": (512, vit_l_16, ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    # "vit_l_32": (224, vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1         ),
    # "vit_h_14": (518, vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1),
    # custom backbone
    # "MRDF_CE": (224, MRDF_CE, None),
    "GAT_video_audio": (128, GAT_video_audio, None),
    "MMTrace": (128, MMTrace, None),
}
# fmt: on


def load_backbone(
    name: str, pretrain: bool = False, *args, **kwargs
) -> Tuple[torch.nn.Module, int, int]:
    input_img_size, model, weights = models[name]
    if pretrain and (weights is not None) and ("weights" not in kwargs):
        kwargs["weights"] = weights
    backbone = model(*args, **kwargs)

    if isinstance(backbone, VisionTransformer):
        feature_size: int = backbone.heads[-1].in_features
        backbone.heads = torch.nn.Identity()  # type: ignore
    elif isinstance(backbone, (ResNet, CifarResNet)):
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()  # type: ignore
    elif isinstance(backbone, Flatten):
        feature_size = input_img_size ** 2
    # elif isinstance(backbone, MRDF_CE):
    #     feature_size = 768
    elif isinstance(backbone, GAT_video_audio):
        feature_size = 256
        video_feature = 192
        audio_feature = 64
        # feature_size = (feature_size, video_feature, audio_feature)
    elif isinstance(backbone, MMTrace):
        feature_size = 384
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return backbone, input_img_size, feature_size


if __name__ == "__main__":

    # for name in models.keys():
    #     backbone, input_img_size, feature_size = load_backbone(name, pretrain=True)
    #     test_img = torch.randn((1, 3, input_img_size, input_img_size))
    #     prototype: torch.Tensor = backbone(test_img)
    #     assert len(prototype.shape) == 2 and prototype.shape[0] == 1
    #     assert feature_size == prototype.shape[1]

    # backbone, input_img_size, feature_size = load_backbone('MRDF_CE', pretrain=False)
    # test_img = torch.randn((1, 3, 100, input_img_size, input_img_size))
    # test_audio = torch.randn((1, 80, 400))
    # mask = torch.BoolTensor(100, 1).fill_(False)
    # # prototype: torch.Tensor = backbone(test_img, test_audio, mask)
    # prototype, v_cross_embeds, a_cross_embeds, v_embeds, a_embeds = backbone(test_img, test_audio, mask)
    # assert len(prototype.shape) == 2 and prototype.shape[0] == 1, prototype.shape
    # assert feature_size == prototype.shape[1], f"{feature_size}{prototype.shape}" 
    # print('test passed')

    backbone, input_img_size, feature_size = load_backbone('GAT_video_audio', pretrain=False)
    test_img = torch.randn((1, 120, input_img_size, input_img_size))
    test_audio = torch.randn((1, 64000))
    video_out, audio_out, fusion_out = backbone(test_img, test_audio)
    assert len(fusion_out.shape) == 2 and fusion_out.shape[0] == 1, fusion_out.shape
    assert feature_size == fusion_out.shape[1], f"{feature_size}{fusion_out.shape}" 
    print('test passed')