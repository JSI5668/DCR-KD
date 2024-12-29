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
from copy import deepcopy

import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings
import torch.utils.model_zoo as model_zoo

from ._base import EncoderMixin

import torch.nn.functional as F
from .unet_parts import *

class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


class ResNetEncoder_with_MultiLevel_Edge(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, edge_channels=1, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool
        self.identity = nn.Identity()
        self.edge_conv1 = nn.Conv2d(edge_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.edge_conv2 = nn.Conv2d(edge_channels, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.edge_conv3 = nn.Conv2d(edge_channels, 512, kernel_size=3, stride=1, padding=1, bias=False)  # 수정된 부분
        self.edge_conv4 = nn.Conv2d(edge_channels, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.edge_conv5 = nn.Conv2d(edge_channels, 2048, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize edge conv layers
        for module in [self.edge_conv1, self.edge_conv2, self.edge_conv3, self.edge_conv4, self.edge_conv5]:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x, edge):
        stages = self.get_stages()

        features = []
        x = self.identity(x)
        features.append(x)

        # Initial conv layers
        x = stages[1][0](x)
        x = stages[1][1](x)
        x = stages[1][2](x)

        # Edge information processing at first level
        edge1 = self.edge_conv1(edge)
        x = x + edge1
        features.append(x)

        # Continue with original model forward pass
        x = stages[2](x)

        # Edge information processing at second level
        edge2 = self.edge_conv2(F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=False))
        x = x + edge2
        features.append(x)

        x = stages[3](x)

        # Edge information processing at third level
        edge3 = self.edge_conv3(F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=False))  # 수정된 부분
        x = x + edge3
        features.append(x)

        x = stages[4](x)
        edge4 = self.edge_conv4(F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=False))
        x = x + edge4
        features.append(x)

        x = stages[5](x)
        edge5 = self.edge_conv5(F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=False))
        x = x + edge5
        features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        # Remove unnecessary keys from state_dict
        keys_to_remove = [
            "fc.bias", "fc.weight", "avgpool.weight", "avgpool.bias"
        ]
        for key in keys_to_remove:
            state_dict.pop(key, None)

        # Remove keys related to layers that are not part of this model
        current_state_dict_keys = list(self.state_dict().keys())
        for key in list(state_dict.keys()):
            if key not in current_state_dict_keys:
                state_dict.pop(key)

        super().load_state_dict(state_dict, strict=False, **kwargs)


class ResNetEncoder_my(ResNet, EncoderMixin):  ## 이게 second_paper_proposed
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool
        self.identity = nn.Identity()
        self.sequential_1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.sequential_2 = nn.Sequential(self.maxpool,self.layer1)

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(256)
        self.CA_2 = ChannelAttention(512)
        self.CA_3 = ChannelAttention(1024)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool,self.layer1),
        ]

    def forward(self, x):

        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)

        features = []
        x = self.identity(x)
        features.append(x)
        x = self.sequential_1(x)
        features.append(x)
        x = self.sequential_2(x)
        features.append(x)

        x_down1 = self.SCM1(x_down1)
        x_down1 = self.down2(x_down1)
        x = torch.cat([x, x_down1], dim=1)
        x = self.Conv1_3(x)
        # x2_CA = self.CA_1(x)  ##
        # x2_ = x * x2_CA       ##
        # x = x + x2_           ##
        x = self.layer2(x)
        features.append(x)

        # x_down2 = self.SCM2(x_down2)
        # x_down2 = self.down3(x_down2)
        # x = torch.cat([x, x_down2], dim=1)
        # x = self.Conv1_4(x)
        # x3_CA = self.CA_2(x)      ##
        # x3_ = x * x3_CA           ##
        # x = x + x3_               ##
        x  = self.layer3(x)
        features.append(x)

        # x_down3 = self.SCM3(x_down3)          ##
        # x_down3 = self.down4(x_down3)         ##
        # x = torch.cat([x, x_down3], dim=1)    ##
        # x = self.Conv1_5(x)                   ##
        # x4_CA = self.CA_3(x)                  ##
        # x4_ = x * x4_CA                       ##
        # x = x + x4_                           ##
        x = self.layer4(x)
        features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)

class ResNetEncoder_my_2(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool
        self.identity = nn.Identity()

        self.sequential_1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.sequential_2 = nn.Sequential(self.maxpool,self.layer1)

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(256)
        self.CA_2 = ChannelAttention(512)
        self.CA_3 = ChannelAttention(1024)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool,self.layer1),
        ]

    def forward(self, x):

        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)

        features = []
        x = self.identity(x)
        features.append(x)
        x = self.sequential_1(x)
        features.append(x)
        x = self.sequential_2(x)  ## X_1,0 (56x56x256)
        features.append(x)

        x_down1 = self.SCM1(x_down1)
        x_down1 = self.down2(x_down1) ## SCM (56x56x256)
        x = torch.cat([x, x_down1], dim=1)
        x = self.Conv1_3(x)
        x2_CA = self.CA_1(x)  ##
        x2_ = x * x2_CA       ##
        x = x + x2_           ##
        x = self.layer2(x)  ##Resblock 2
        features.append(x)

        x_down2 = self.SCM2(x_down2)
        x_down2 = self.down3(x_down2)
        x = torch.cat([x, x_down2], dim=1)
        x = self.Conv1_4(x)
        x3_CA = self.CA_2(x)      ##
        x3_ = x * x3_CA           ##
        x = x + x3_               ##
        x  = self.layer3(x)
        features.append(x)

        x_down3 = self.SCM3(x_down3)          ##
        x_down3 = self.down4(x_down3)         ##
        x = torch.cat([x, x_down3], dim=1)    ##
        x = self.Conv1_5(x)                   ##
        x4_CA = self.CA_3(x)                  ##
        x4_ = x * x4_CA                       ##
        x = x + x4_                           ##
        x = self.layer4(x)
        features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)


class ResNetEncoder_my_2_with_MultiLevel_Edge(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, edge_channels=1, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool
        self.identity = nn.Identity()

        # self.edge_conv1 = nn.Conv2d(edge_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.edge_conv2 = nn.Conv2d(edge_channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.edge_conv3 = nn.Conv2d(edge_channels, 512, kernel_size=3, stride=1, padding=1, bias=False)

        self.sequential_1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.sequential_2 = nn.Sequential(self.maxpool, self.layer1)

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(256)
        self.CA_2 = ChannelAttention(512)
        self.CA_3 = ChannelAttention(1024)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
        ]

    def forward(self, x, edge):
        x_down1 = F.interpolate(edge, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(edge, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(edge, scale_factor=0.125, mode='bilinear', align_corners=False)

        features = []
        x = self.identity(x)
        features.append(x)
        x = self.sequential_1(x)

        # Edge information processing at first level
        # edge1 = self.edge_conv1(edge)
        # x = x + edge1
        features.append(x)

        x = self.sequential_2(x)  ## X_1,0 (56x56x256)
        features.append(x)

        # Edge information processing at second level
        x_down1 = self.edge_conv2(x_down1) ## GPT
        x_down1 = self.SCM1(x_down1)
        x_down1 = self.down2(x_down1)  ## SCM (56x56x256)
        # x = torch.cat([x, x_down1, edge2], dim=1) ## GPT
        x = torch.cat([x, x_down1], dim=1)
        x = self.Conv1_3(x)
        x2_CA = self.CA_1(x)  ##
        x2_ = x * x2_CA  ##
        x = x + x2_  ##
        x = self.layer2(x)  ##Resblock 2
        features.append(x)

        # Edge information processing at third level
        x_down2 = self.edge_conv2(x_down2)
        x_down2 = self.SCM2(x_down2)
        x_down2 = self.down3(x_down2)
        # x = torch.cat([x, x_down2, edge3], dim=1)
        x = torch.cat([x, x_down2], dim=1)
        x = self.Conv1_4(x)
        x3_CA = self.CA_2(x)  ##
        x3_ = x * x3_CA  ##
        x = x + x3_  ##
        x = self.layer3(x)
        features.append(x)

        # Edge information processing at fourth level
        x_down3 = self.edge_conv2(x_down3)
        x_down3 = self.SCM3(x_down3)  ##
        x_down3 = self.down4(x_down3)  ##
        # x = torch.cat([x, x_down3, edge4], dim=1)  ##
        x = torch.cat([x, x_down3], dim=1)  ##
        x = self.Conv1_5(x)  ##
        x4_CA = self.CA_3(x)  ##
        x4_ = x * x4_CA  ##
        x = x + x4_  ##
        x = self.layer4(x)
        features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        # Remove unnecessary keys from state_dict
        keys_to_remove = [
            "fc.bias", "fc.weight", "avgpool.weight", "avgpool.bias"
        ]
        for key in keys_to_remove:
            state_dict.pop(key, None)

        # Remove keys related to layers that are not part of this model
        current_state_dict_keys = list(self.state_dict().keys())
        for key in list(state_dict.keys()):
            if key not in current_state_dict_keys:
                state_dict.pop(key)

        # Initialize edge conv layers
        # for module in [self.edge_conv1, self.edge_conv2, self.edge_conv3]:
        #     nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

        super().load_state_dict(state_dict, strict=False, **kwargs)

class ResNetEncoder_my_2_input4channels_edgergbconcat(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, block=BasicBlock, layers=[3, 4, 6, 3], pretrained_url=None, **kwargs):
        super().__init__(block, layers, **kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 4

        del self.fc
        del self.avgpool
        self.identity = nn.Identity()

        # 첫 번째 레이어를 4채널 입력을 받을 수 있도록 수정
        original_conv1 = self.conv1
        self.conv1 = nn.Conv2d(4, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                               stride=original_conv1.stride, padding=original_conv1.padding, bias=original_conv1.bias)

        # 나머지 레이어는 그대로 유지
        self.sequential_1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.sequential_2 = nn.Sequential(self.maxpool, self.layer1)

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(129)
        self.SCM2 = SCM(257)
        self.SCM3 = SCM(513)

        self.CA_1 = ChannelAttention(256)
        self.CA_2 = ChannelAttention(512)
        self.CA_3 = ChannelAttention(1024)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 사전 학습된 가중치 로드
        if pretrained_url:
            self._load_pretrained_weights(pretrained_url)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
        ]

    def forward(self, x):
        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)

        features = []
        x = self.identity(x)
        features.append(x)
        x = self.sequential_1(x)
        features.append(x)
        x = self.sequential_2(x)
        features.append(x)

        x_down1 = self.SCM1(x_down1)
        x_down1 = self.down2(x_down1)
        x = torch.cat([x, x_down1], dim=1)
        x = self.Conv1_3(x)
        x2_CA = self.CA_1(x)
        x2_ = x * x2_CA
        x = x + x2_
        x = self.layer2(x)
        features.append(x)

        x_down2 = self.SCM2(x_down2)
        x_down2 = self.down3(x_down2)
        x = torch.cat([x, x_down2], dim=1)
        x = self.Conv1_4(x)
        x3_CA = self.CA_2(x)
        x3_ = x * x3_CA
        x = x + x3_
        x = self.layer3(x)
        features.append(x)

        x_down3 = self.SCM3(x_down3)
        x_down3 = self.down4(x_down3)
        x = torch.cat([x, x_down3], dim=1)
        x = self.Conv1_5(x)
        x4_CA = self.CA_3(x)
        x4_ = x * x4_CA
        x = x + x4_
        x = self.layer4(x)
        features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)


class ResNetEncoder_input4channel_edgergbconcat(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 4

        del self.fc
        del self.avgpool

        # 첫 번째 레이어를 4채널 입력을 받을 수 있도록 수정
        original_conv1 = self.conv1
        self.conv1 = nn.Conv2d(4, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                               stride=original_conv1.stride, padding=original_conv1.padding, bias=original_conv1.bias)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)

class ResNetEncoder_my_ablation_case2(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool
        self.identity = nn.Identity()
        self.sequential_1 = nn.Sequential(self.conv1, self.bn1, self.relu)
        self.sequential_2 = nn.Sequential(self.maxpool,self.layer1)

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(256)
        self.CA_2 = ChannelAttention(512)
        self.CA_3 = ChannelAttention(1024)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool,self.layer1),
        ]

    def forward(self, x):

        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)

        features = []
        x = self.identity(x)
        features.append(x)
        x = self.sequential_1(x)
        features.append(x)
        x = self.sequential_2(x)
        features.append(x)

        x_down1 = self.SCM1(x_down1)
        x_down1 = self.down2(x_down1)
        x = torch.cat([x, x_down1], dim=1)
        x = self.Conv1_3(x)
        # x2_CA = self.CA_1(x)  ##
        # x2_ = x * x2_CA       ##
        # x = x + x2_           ##
        x = self.layer2(x)
        features.append(x)

        x_down2 = self.SCM2(x_down2)
        x_down2 = self.down3(x_down2)
        x = torch.cat([x, x_down2], dim=1)
        x = self.Conv1_4(x)
        # x3_CA = self.CA_2(x)      ##
        # x3_ = x * x3_CA           ##
        # x = x + x3_               ##
        x  = self.layer3(x)
        features.append(x)

        x_down3 = self.SCM3(x_down3)          ##
        x_down3 = self.down4(x_down3)         ##
        x = torch.cat([x, x_down3], dim=1)    ##
        x = self.Conv1_5(x)                   ##
        # x4_CA = self.CA_3(x)                  ##
        # x4_ = x * x4_CA                       ##
        # x = x + x4_                           ##
        x = self.layer4(x)
        features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)

new_settings = {
    "resnet18": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth",  # noqa
    },
    "resnet50": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth",  # noqa
    },
    "resnext50_32x4d": {
        "imagenet": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth",  # noqa
    },
    "resnext101_32x4d": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth",  # noqa
    },
    "resnext101_32x8d": {
        "imagenet": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth",  # noqa
    },
    "resnext101_32x16d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth",  # noqa
    },
    "resnext101_32x32d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
    },
    "resnext101_32x48d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
    },
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }


resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet18"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet34"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        # "encoder": ResNetEncoder_with_MultiLevel_Edge,  ## 두 번째 논문은 ResNetEncoder_my_2
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet101"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet152"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext50_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x8d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x16d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x32d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x48d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}
