""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torchvision.transforms as transforms
import torch.nn.functional as F
tf = transforms.ToPILImage()


"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`
Attributes:
    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)
Methods:
    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).
        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)
        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
# from copy import deepcopy
#
# import torch.nn as nn
#
# from torchvision.models.resnet import ResNet
# from torchvision.models.resnet import BasicBlock
# from torchvision.models.resnet import Bottleneck
# from pretrainedmodels.models.torchvision_models import pretrained_settings
#
# from ._base import EncoderMixin
#
#
# class ResNetEncoder(ResNet, EncoderMixin):
#     def __init__(self, out_channels, depth=5, **kwargs):
#         super().__init__(**kwargs)
#         self._depth = depth
#         self._out_channels = out_channels
#         self._in_channels = 3
#
#         del self.fc
#         del self.avgpool
#
#     def get_stages(self):
#         return [
#             nn.Identity(),
#             nn.Sequential(self.conv1, self.bn1, self.relu),
#             nn.Sequential(self.maxpool, self.layer1),
#             self.layer2,
#             self.layer3,
#             self.layer4,
#         ]
#
#     def forward(self, x):
#         stages = self.get_stages()
#
#         features = []
#         for i in range(self._depth + 1):
#             x = stages[i](x)
#             features.append(x)
#
#         return features
#
#     def load_state_dict(self, state_dict, **kwargs):
#         state_dict.pop("fc.bias", None)
#         state_dict.pop("fc.weight", None)
#         super().load_state_dict(state_dict, **kwargs)
#
#
# new_settings = {
#     "resnet18": {
#         "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",
#         # noqa
#         "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth",
#         # noqa
#     },
#     "resnet50": {
#         "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",
#         # noqa
#         "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth",
#         # noqa
#     },
#     "resnext50_32x4d": {
#         "imagenet": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
#         "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth",
#         # noqa
#         "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth",
#         # noqa
#     },
#     "resnext101_32x4d": {
#         "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth",
#         # noqa
#         "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth",
#         # noqa
#     },
#     "resnext101_32x8d": {
#         "imagenet": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
#         "instagram": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
#         "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth",
#         # noqa
#         "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth",
#         # noqa
#     },
#     "resnext101_32x16d": {
#         "instagram": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
#         "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth",
#         # noqa
#         "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth",
#         # noqa
#     },
#     "resnext101_32x32d": {
#         "instagram": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
#     },
#     "resnext101_32x48d": {
#         "instagram": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
#     },
# }
#
# pretrained_settings = deepcopy(pretrained_settings)
# for model_name, sources in new_settings.items():
#     if model_name not in pretrained_settings:
#         pretrained_settings[model_name] = {}
#
#     for source_name, source_url in sources.items():
#         pretrained_settings[model_name][source_name] = {
#             "url": source_url,
#             "input_size": [3, 224, 224],
#             "input_range": [0, 1],
#             "mean": [0.485, 0.456, 0.406],
#             "std": [0.229, 0.224, 0.225],
#             "num_classes": 1000,
#         }
#
# resnet_encoders = {
#     "resnet18": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet18"],
#         "params": {
#             "out_channels": (3, 64, 64, 128, 256, 512),
#             "block": BasicBlock,
#             "layers": [2, 2, 2, 2],
#         },
#     },
#     "resnet34": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet34"],
#         "params": {
#             "out_channels": (3, 64, 64, 128, 256, 512),
#             "block": BasicBlock,
#             "layers": [3, 4, 6, 3],
#         },
#     },
#     "resnet50": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet50"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 6, 3],
#         },
#     },
#     "resnet101": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet101"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#         },
#     },
#     "resnet152": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet152"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 8, 36, 3],
#         },
#     },
#     "resnext50_32x4d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnext50_32x4d"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 6, 3],
#             "groups": 32,
#             "width_per_group": 4,
#         },
#     },
#     "resnext101_32x4d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnext101_32x4d"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 4,
#         },
#     },
#     "resnext101_32x8d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnext101_32x8d"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 8,
#         },
#     },
#     "resnext101_32x16d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnext101_32x16d"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 16,
#         },
#     },
#     "resnext101_32x32d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnext101_32x32d"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 32,
#         },
#     },
#     "resnext101_32x48d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnext101_32x48d"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 48,
#         },
#     },
# }

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_attnetion_decoder_concat(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_attnetion_decoder_concat, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(3072, 1536 // factor, bilinear)
        self.up2 = Up(1536, 768 // factor, bilinear)
        self.up3 = Up(768, 384 // factor, bilinear)
        self.up4 = Up(384, 192, bilinear)
        self.outc = OutConv(192, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

    def forward(self, x, outputs_pre):
        x1 = self.inc(x)
        x1_attention_0 = x1 * outputs_pre[:,0,:,:].unsqueeze(dim=1)
        x1_attention_1 = x1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x1_concat = torch.cat([x1, x1_attention_0, x1_attention_1], dim=1)

        x2 = self.down1(x1)
        x2_attention_0 = self.down1(x1_attention_0)
        x2_attention_1 = self.down1(x1_attention_1)
        x2_concat = torch.cat([x2, x2_attention_0, x2_attention_1], dim=1)

        x3 = self.down2(x2)
        x3_attention_0 = self.down2(x2_attention_0)
        x3_attention_1 = self.down2(x2_attention_1)
        x3_concat = torch.cat([x3, x3_attention_0, x3_attention_1], dim=1)

        x4 = self.down3(x3)
        x4_attention_0 = self.down3(x3_attention_0)
        x4_attention_1 = self.down3(x3_attention_1)
        x4_concat = torch.cat([x4, x4_attention_0, x4_attention_1], dim=1)

        x5 = self.down4(x4)
        x5_attention_0 = self.down4(x4_attention_0)
        x5_attention_1 = self.down4(x4_attention_1)
        x5_concat = torch.cat([x5, x5_attention_0, x5_attention_1], dim=1)

        x = self.up1(x5_concat, x4_concat)
        x = self.up2(x, x3_concat)
        x = self.up3(x, x2_concat)
        x = self.up4(x, x1_concat)
        logits = self.outc(x)
        return logits


class UNet_attnetion_encoder_concat(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_attnetion_encoder_concat, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

    def forward(self, x, outputs_pre):
        x1 = self.inc(x)
        # x1_attention_0 = x1 * outputs_pre[:,0,:,:].unsqueeze(dim=1)
        x1_attention_1 = x1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x1_concat = torch.cat([x1, x1_attention_1], dim=1)
        x1_concat = self.Conv1_1(x1_concat)

        x2 = self.down1(x1_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x2.shape[-2:], mode='nearest')
        x2_attention_1 = x2 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x2_concat = torch.cat([x2, x2_attention_1], dim=1)
        x2_concat = self.Conv1_2(x2_concat)

        x3 = self.down2(x2_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x3.shape[-2:], mode='nearest')
        x3_attention_1 = x3 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x3_concat = torch.cat([x3, x3_attention_1], dim=1)
        x3_concat = self.Conv1_3(x3_concat)

        x4 = self.down3(x3_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x4.shape[-2:], mode='nearest')
        x4_attention_1 = x4 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x4_concat = torch.cat([x4, x4_attention_1], dim=1)
        x4_concat = self.Conv1_4(x4_concat)

        x5 = self.down4(x4_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x5.shape[-2:], mode='nearest')
        x5_attention_1 = x5 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x5_concat = torch.cat([x5, x5_attention_1], dim=1)
        x5_concat = self.Conv1_4(x5_concat)

        x = self.up1(x5_concat, x4_concat)
        x = self.up2(x, x3_concat)
        x = self.up3(x, x2_concat)
        x = self.up4(x, x1_concat)
        logits = self.outc(x)
        return logits



class UNet_attnetion_encoder_each_sum(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_attnetion_encoder_each_sum, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

    def forward(self, x, outputs_pre):
        x1 = self.inc(x)
        x1_attention_0 = x1 * outputs_pre[:,0,:,:].unsqueeze(dim=1)
        x1_attention_1 = x1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x1 = 0.5 * x1_attention_0 + 1.5 * x1_attention_1

        x2 = self.down1(x1)
        outputs_pre = F.interpolate(outputs_pre, size=x2.shape[-2:], mode='nearest')
        x2_attention_0 = x2 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x2_attention_1 = x2 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x2 = 1.5 * x2_attention_0 + 0.5 * x2_attention_1

        x3 = self.down2(x2)
        outputs_pre = F.interpolate(outputs_pre, size=x3.shape[-2:], mode='nearest')
        x3_attention_0 = x3 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x3_attention_1 = x3 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x3 = 0.5 * x3_attention_0 + 1.5 * x3_attention_1

        x4 = self.down3(x3)
        outputs_pre = F.interpolate(outputs_pre, size=x4.shape[-2:], mode='nearest')
        x4_attention_0 = x4 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x4_attention_1 = x4 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x4 = 1.5 * x4_attention_0 + 0.5 * x4_attention_1

        x5 = self.down4(x4)
        outputs_pre = F.interpolate(outputs_pre, size=x5.shape[-2:], mode='nearest')
        x5_attention_0 = x5 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x5_attention_1 = x5 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x4 = 0.5 * x5_attention_0 + 1.5 * x5_attention_1

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_attnetion_encoder_decoder_concat(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_attnetion_encoder_decoder_concat, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc_new = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

    def forward(self, x, outputs_pre):
        x1 = self.inc(x)
        # x1_attention_0 = x1 * outputs_pre[:,0,:,:].unsqueeze(dim=1)
        x1_attention_1 = x1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x1_concat = torch.cat([x1, x1_attention_1], dim=1)
        x1_concat = self.Conv1_1(x1_concat)

        x2 = self.down1(x1_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x2.shape[-2:], mode='nearest')
        x2_attention_1 = x2 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x2_concat = torch.cat([x2, x2_attention_1], dim=1)
        x2_concat = self.Conv1_2(x2_concat)

        x3 = self.down2(x2_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x3.shape[-2:], mode='nearest')
        x3_attention_1 = x3 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x3_concat = torch.cat([x3, x3_attention_1], dim=1)
        x3_concat = self.Conv1_3(x3_concat)

        x4 = self.down3(x3_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x4.shape[-2:], mode='nearest')
        x4_attention_1 = x4 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x4_concat = torch.cat([x4, x4_attention_1], dim=1)
        x4_concat = self.Conv1_4(x4_concat)

        x5 = self.down4(x4_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x5.shape[-2:], mode='nearest')
        x5_attention_1 = x5 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)
        x5_concat = torch.cat([x5, x5_attention_1], dim=1)
        x5_concat = self.Conv1_4(x5_concat)  ## if bilinear true
        # x5_concat = self.Conv1_5(x5_concat)  ## if bilinear false

        x = self.up1(x5_concat, x4_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x.shape[-2:], mode='nearest')
        x1_decoder_attention_0 = x * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x = torch.cat([x, x1_decoder_attention_0], dim=1)
        x = self.Conv1_3(x)
        # x = self.Conv1_4(x)

        x = self.up2(x, x3_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x.shape[-2:], mode='nearest')
        x2_decoder_attention_0 = x * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x = torch.cat([x, x2_decoder_attention_0], dim=1)
        x = self.Conv1_2(x)
        # x = self.Conv1_3(x)

        x = self.up3(x, x2_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x.shape[-2:], mode='nearest')
        x3_decoder_attention_0 = x * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x = torch.cat([x, x3_decoder_attention_0], dim=1)
        x = self.Conv1_1(x)
        # x = self.Conv1_2(x)

        x = self.up4(x, x1_concat)
        outputs_pre = F.interpolate(outputs_pre, size=x.shape[-2:], mode='nearest')
        x4_decoder_attention_0 = x * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x = torch.cat([x, x4_decoder_attention_0], dim=1)
        x = self.Conv1_1(x)

        logits = self.outc_new(x)
        return logits


class UNet_double_encoder_decoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_double_encoder_decoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

    def forward(self, x, outputs_pre):
        x1 = self.inc(x)
        x1_attention_0 = x1 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x1_attention_1 = x1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)

        x2_0 = self.down1(x1_attention_0)
        x2_1 = self.down1(x1_attention_1)
        outputs_pre = F.interpolate(outputs_pre, size=x2_0.shape[-2:], mode='nearest')
        x2_attention_0 = x2_0 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x2_attention_1 = x2_1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)

        x3_0 = self.down2(x2_attention_0)
        x3_1 = self.down2(x2_attention_1)
        outputs_pre = F.interpolate(outputs_pre, size=x3_0.shape[-2:], mode='nearest')
        x3_attention_0 = x3_0 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x3_attention_1 = x3_1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)

        x4_0 = self.down3(x3_attention_0)
        x4_1 = self.down3(x3_attention_1)
        outputs_pre = F.interpolate(outputs_pre, size=x4_0.shape[-2:], mode='nearest')
        x4_attention_0 = x4_0 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x4_attention_1 = x4_1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)

        x5_0 = self.down4(x4_attention_0)
        x5_1 = self.down4(x4_attention_1)
        outputs_pre = F.interpolate(outputs_pre, size=x5_0.shape[-2:], mode='nearest')
        # x5_attention_0 = x5_0 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        # x5_attention_1 = x5_1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)

        x_0 = self.up1(x5_0, x4_0)
        x_1 = self.up1(x5_1, x4_1)
        outputs_pre = F.interpolate(outputs_pre, size=x_0.shape[-2:], mode='nearest')
        x1_decoder_attention_0 = x_0 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x1_decoder_attention_1 = x_1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)

        x_0 = self.up2(x1_decoder_attention_0, x3_0)
        x_1 = self.up2(x1_decoder_attention_1, x3_1)
        outputs_pre = F.interpolate(outputs_pre, size=x_0.shape[-2:], mode='nearest')
        x2_decoder_attention_0 = x_0 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x2_decoder_attention_1 = x_1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)

        x_0 = self.up3(x2_decoder_attention_0, x2_0)
        x_1 = self.up3(x2_decoder_attention_1, x2_1)
        outputs_pre = F.interpolate(outputs_pre, size=x_0.shape[-2:], mode='nearest')
        x3_decoder_attention_0 = x_0 * outputs_pre[:, 0, :, :].unsqueeze(dim=1)
        x3_decoder_attention_1 = x_1 * outputs_pre[:, 1, :, :].unsqueeze(dim=1)

        x_0 = self.up4(x3_decoder_attention_0, x1)
        x_1 = self.up4(x3_decoder_attention_1, x1)
        x = torch.cat([x_0, x_1], dim=1)
        x = self.Conv1_1(x)

        logits = self.outc(x)
        return logits



class UNet_double_en_oneattention(nn.Module):        ## 12/16 input image 2 개, 하나에만 attention
    def __init__(self, n_channels, n_classes, n_classes_smallbranch, bilinear=False):
        super(UNet_double_en_oneattention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_classes_smallbranch = n_classes_smallbranch
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.outc_smallbranch = OutConv(64, n_classes_smallbranch)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

    def forward(self, x, outputs_pre):
        x_attention = x * outputs_pre[:,1,:,:].unsqueeze(dim=1)   ## input 에다가 소수 클래스 채널 attention
        # tf(x_attention[0]).show()
        x1 = self.inc(x)
        x1_attention = self.inc(x_attention)
        x1_attention_detach = x1_attention.clone().detach()
        x1 = x1 * nn.Sigmoid()(x1_attention_detach)

        x2 = self.down1(x1)
        x2_attention = self.down1(x1_attention)
        x2_attention_detach = x2_attention.clone().detach()
        x2 = x2 * nn.Sigmoid()(x2_attention_detach)

        x3 = self.down2(x2)
        x3_attention = self.down2(x2_attention)
        x3_attention_detach = x3_attention.clone().detach()
        x3 = x3 * nn.Sigmoid()(x3_attention_detach)

        x4 = self.down3(x3)
        x4_attention = self.down3(x3_attention)
        x4_attention_detach = x4_attention.clone().detach()
        x4 = x4 * nn.Sigmoid()(x4_attention_detach)

        x5 = self.down4(x4)
        x5_attention = self.down4(x4_attention)
        x5_attention_detach = x5_attention.clone().detach()
        x5 = x5 * nn.Sigmoid()(x5_attention_detach)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_smallclasses_branch = self.up1(x5_attention, x4_attention)
        x_smallclasses_branch = self.up2(x_smallclasses_branch, x3_attention)
        x_smallclasses_branch = self.up3(x_smallclasses_branch, x2_attention)
        x_smallclasses_branch = self.up4(x_smallclasses_branch, x1_attention)

        logits = self.outc(x)
        logits_small_branch = self.outc_smallbranch(x_smallclasses_branch)
        return logits, logits_small_branch


class UNet_my(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_my, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

    def forward(self, x):
        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
        x_down4 = F.interpolate(x, scale_factor=0.0625, mode='bilinear', align_corners=False)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x_down1 = self.SCM1(x_down1)
        x2 = torch.cat([x2, x_down1], dim=1)
        x2 = self.Conv1_2(x2)

        x3 = self.down2(x2)
        x_down2 = self.SCM2(x_down2)
        x3 = torch.cat([x3, x_down2], dim=1)
        x3 = self.Conv1_3(x3)

        x4 = self.down3(x3)
        x_down3 = self.SCM3(x_down3)
        x4 = torch.cat([x4, x_down3], dim=1)
        x4 = self.Conv1_4(x4)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class UNet_chae_pretrained(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_chae_pretrained, self).__init__()
        self.enc = ResNetEncoder(n_channels, bilinear=False)
        self.dec = decoder_my_2(n_classes, bilinear=False)

    def forward(self, x):
        enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5 = self.enc(x)
        dec_out = self.dec(enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5)

        return enc_out_5, dec_out

class UNet_chae(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_chae, self).__init__()
        self.enc = encoder_my_2(n_channels, bilinear=False)
        self.dec = decoder_my_2(n_classes, bilinear=False)

    def forward(self, x):
        enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5 = self.enc(x)
        dec_out = self.dec(enc_out_1, enc_out_2, enc_out_3, enc_out_4, enc_out_5)

        return dec_out
class decoder_my_2(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(decoder_my_2, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(128)
        self.CA_2 = ChannelAttention(256)
        self.CA_3 = ChannelAttention(512)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
class encoder_my_2(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(encoder_my_2, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(128)
        self.CA_2 = ChannelAttention(256)
        self.CA_3 = ChannelAttention(512)

    def forward(self, x):
        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
        x_down4 = F.interpolate(x, scale_factor=0.0625, mode='bilinear', align_corners=False)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x_down1 = self.SCM1(x_down1)
        x2 = torch.cat([x2, x_down1], dim=1)
        x2 = self.Conv1_2(x2)
        x2_CA = self.CA_1(x2)
        x2_ = x2 * x2_CA
        x2 = x2 + x2_

        x3 = self.down2(x2)
        x_down2 = self.SCM2(x_down2)
        x3 = torch.cat([x3, x_down2], dim=1)
        x3 = self.Conv1_3(x3)
        x3_CA = self.CA_2(x3)
        x3_ = x3 * x3_CA
        x3 = x3 + x3_

        x4 = self.down3(x3)
        x_down3 = self.SCM3(x_down3)
        x4 = torch.cat([x4, x_down3], dim=1)
        x4 = self.Conv1_4(x4)
        x4_CA = self.CA_3(x4)
        x4_ = x4 * x4_CA
        x4 = x4 + x4_

        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5
class UNet_my_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_my_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(128)
        self.CA_2 = ChannelAttention(256)
        self.CA_3 = ChannelAttention(512)

    def forward(self, x):
        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
        x_down4 = F.interpolate(x, scale_factor=0.0625, mode='bilinear', align_corners=False)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x_down1 = self.SCM1(x_down1)
        x2 = torch.cat([x2, x_down1], dim=1)
        x2 = self.Conv1_2(x2)
        x2_CA = self.CA_1(x2)
        x2_ = x2 * x2_CA
        x2 = x2 + x2_

        x3 = self.down2(x2)
        x_down2 = self.SCM2(x_down2)
        x3 = torch.cat([x3, x_down2], dim=1)
        x3 = self.Conv1_3(x3)
        x3_CA = self.CA_2(x3)
        x3_ = x3 * x3_CA
        x3 = x3 + x3_

        x4 = self.down3(x3)
        x_down3 = self.SCM3(x_down3)
        x4 = torch.cat([x4, x_down3], dim=1)
        x4 = self.Conv1_4(x4)
        x4_CA = self.CA_3(x4)
        x4_ = x4 * x4_CA
        x4 = x4 + x4_

        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # return logits, x2, x3, x4, x5
        return logits

##x2 : 첫 번째 encoder block, x3 : 두 번째 encoder block (fusion 되기 전), x4: 세 번째, x5


class EdgeNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(EdgeNet, self).__init__()
        self.Conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1, bias=False)
        self.Conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.Conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.Conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)

        # self.outc = OutConv(32, n_classes)
        self.outc = OutConv(64, n_classes)

        self.BN1 = nn.BatchNorm2d(16)
        self.BN2 = nn.BatchNorm2d(32)
        self.BN3 = nn.BatchNorm2d(64)
        self.BN4 = nn.BatchNorm2d(128)

        self.ReLU = nn.ReLU(inplace=True)


    def forward(self, x):
        x1 = self.Conv1(x)
        x1 = self.BN1(x1)
        x1 = self.ReLU(x1)

        x2 = self.Conv2(x1)
        x2 = self.BN2(x2)
        x2 = self.ReLU(x2)

        x3 = self.Conv3(x2)
        self.BN3 = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU(inplace=True)
        #
        # x4 = self.Conv4(x3)
        # self.BN4 = nn.BatchNorm2d(128)
        # self.ReLU = nn.ReLU(inplace=True)

        # logits = self.outc(x2)
        logits = self.outc(x3)

        return logits


class UNet_my_3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_my_3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.outc_1 = OutConv(256, n_classes)
        self.outc_2 = OutConv(128, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(128)
        self.CA_2 = ChannelAttention(256)
        self.CA_3 = ChannelAttention(512)

    def forward(self, x):
        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
        x_down4 = F.interpolate(x, scale_factor=0.0625, mode='bilinear', align_corners=False)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x_down1 = self.SCM1(x_down1)
        x2 = torch.cat([x2, x_down1], dim=1)
        x2 = self.Conv1_2(x2)
        x2_CA = self.CA_1(x2)
        x2_ = x2 * x2_CA
        x2 = x2 + x2_

        x3 = self.down2(x2)
        x_down2 = self.SCM2(x_down2)
        x3 = torch.cat([x3, x_down2], dim=1)
        x3 = self.Conv1_3(x3)
        x3_CA = self.CA_2(x3)
        x3_ = x3 * x3_CA
        x3 = x3 + x3_

        x4 = self.down3(x3)
        x_down3 = self.SCM3(x_down3)
        x4 = torch.cat([x4, x_down3], dim=1)
        x4 = self.Conv1_4(x4)
        x4_CA = self.CA_3(x4)
        x4_ = x4 * x4_CA
        x4 = x4 + x4_

        x5 = self.down4(x4)

        x = self.up1(x5, x4)  ## 256
        x_out_1 = self.outc_1(x)

        x = self.up2(x, x3)   ## 128
        x_out_2 = self.outc_2(x)

        x = self.up3(x, x2)   ## 64
        x_out_3 = self.outc(x)

        x = self.up4(x, x1)   ## 64
        logits = self.outc(x)
        return logits, x_out_1, x_out_2, x_out_3


class W_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(W_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(128)
        self.CA_2 = ChannelAttention(256)
        self.CA_3 = ChannelAttention(512)

############################################################
        self.second_inc = DoubleConv(64, 64)
        self.second_down1 = Down(64, 128)
        self.second_down2 = Down(128, 256)
        self.second_down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.second_down4 = Down(512, 1024 // factor)

        self.second_up1 = Up(1024, 512 // factor, bilinear)
        self.second_up2 = Up(512, 256 // factor, bilinear)
        self.second_up3 = Up(256, 128 // factor, bilinear)
        self.second_up4 = Up(128, 64, bilinear)

        self.second_Conv1_1 = Conv1x1(64, 128)
        self.second_Conv1_2 = Conv1x1(128, 256)
        self.second_Conv1_3 = Conv1x1(256, 512)
        self.second_Conv1_4 = Conv1x1(1024, 512)
        self.second_Conv1_5 = Conv1x1(2048, 1024)

        self.second_SCM1 = SCM(128)
        self.second_SCM2 = SCM(256)
        self.second_SCM3 = SCM(512)

        self.second_CA_1 = ChannelAttention(64)
        self.second_CA_2 = ChannelAttention(128)
        self.second_CA_3 = ChannelAttention(256)
        self.second_CA_4 = ChannelAttention(512)

        self.second_outc = OutConv(64, n_classes)
    def forward(self, x):
        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
        x_down4 = F.interpolate(x, scale_factor=0.0625, mode='bilinear', align_corners=False)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x_down1 = self.SCM1(x_down1)
        x2 = torch.cat([x2, x_down1], dim=1)
        x2 = self.Conv1_2(x2)
        x2_CA = self.CA_1(x2)
        x2_ = x2 * x2_CA
        x2 = x2 + x2_

        x3 = self.down2(x2)
        x_down2 = self.SCM2(x_down2)
        x3 = torch.cat([x3, x_down2], dim=1)
        x3 = self.Conv1_3(x3)
        x3_CA = self.CA_2(x3)
        x3_ = x3 * x3_CA
        x3 = x3 + x3_

        x4 = self.down3(x3)
        x_down3 = self.SCM3(x_down3)
        x4 = torch.cat([x4, x_down3], dim=1)
        x4 = self.Conv1_4(x4)
        x4_CA = self.CA_3(x4)
        x4_ = x4 * x4_CA
        x4 = x4 + x4_

        x5 = self.down4(x4)

        x1_decoder = self.up1(x5, x4)
        x2_decoder = self.up2(x1_decoder, x3)
        x3_decoder = self.up3(x2_decoder, x2)
        x4_decoder = self.up4(x3_decoder, x1)

        ####Second UNet
        x4_decoder_CA = self.second_CA_1(x4_decoder)
        second_x1_CA = x4_decoder * x4_decoder_CA
        second_x1 = x4_decoder + second_x1_CA

        second_x1 = self.second_inc(second_x1)
        ##second_x1 = self.second_inc(x4_decoder)

        second_x2 = self.second_down1(second_x1)
        x3_decoder = self.second_Conv1_1(x3_decoder)
        x3_decoder_CA = self.second_CA_2(x3_decoder)
        second_x2_CA = second_x2 * x3_decoder_CA
        second_x2 = second_x2 + second_x2_CA

        second_x3 = self.second_down2(second_x2)
        x2_decoder = self.second_Conv1_2(x2_decoder)
        x2_decoder_CA = self.second_CA_3(x2_decoder)
        second_x3_CA = second_x3 * x2_decoder_CA
        second_x3 = second_x3 + second_x3_CA

        second_x4 = self.second_down3(second_x3)
        x1_decoder = self.second_Conv1_3(x1_decoder)
        x1_decoder_CA = self.second_CA_4(x1_decoder)
        second_x4_CA = second_x4 * x1_decoder_CA
        second_x4 = second_x4 + second_x4_CA

        second_x5 = self.second_down4(second_x4)

        second_x1_decoder = self.second_up1(second_x5, second_x4)
        second_x2_decoder = self.second_up2(second_x1_decoder, second_x3)
        second_x3_decoder = self.second_up3(second_x2_decoder, second_x2)
        second_x4_decoder = self.second_up4(second_x3_decoder, second_x1)

        logits = self.second_outc(second_x4_decoder)

        # logits = self.outc(x4_decoder)
        return logits