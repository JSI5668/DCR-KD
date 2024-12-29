import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):  # Edge decoder의 각 디코더 블록의 feature map을 반환
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # 동일한 공간 해상도를 가진 첫 번째 스킵 제거
        features = features[::-1]  # 인코더의 헤드에서 시작하도록 채널을 역순으로 변경
        dense_x = {}
        edge_feature_maps = []  # 반환할 edge decoder의 feature maps

        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features
                    )
                edge_feature_maps.append(dense_x[f"x_{depth_idx}_{depth_idx}"])  # Store edge feature maps

        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth - 1}"])
        edge_feature_maps.append(dense_x[f"x_{0}_{self.depth}"])  # Store final edge feature map

        return dense_x[f"x_{0}_{self.depth}"], edge_feature_maps  # Return final output and edge feature maps



# class UnetPlusPlusDecoder(nn.Module):
#     def __init__(
#         self,
#         encoder_channels,
#         decoder_channels,
#         n_blocks=5,
#         use_batchnorm=True,
#         attention_type=None,
#         center=False,
#     ):
#         super().__init__()
#
#         if n_blocks != len(decoder_channels):
#             raise ValueError(
#                 "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
#                     n_blocks, len(decoder_channels)
#                 )
#             )
#
#         # remove first skip with same spatial resolution
#         encoder_channels = encoder_channels[1:]
#         # reverse channels to start from head of encoder
#         encoder_channels = encoder_channels[::-1]
#
#         # computing blocks input and output channels
#         head_channels = encoder_channels[0]
#         self.in_channels = [head_channels] + list(decoder_channels[:-1])
#         self.skip_channels = list(encoder_channels[1:]) + [0]
#         self.out_channels = decoder_channels
#         if center:
#             self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
#         else:
#             self.center = nn.Identity()
#
#         # combine decoder keyword arguments
#         kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
#
#         blocks = {}
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(layer_idx + 1):
#                 if depth_idx == 0:
#                     in_ch = self.in_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
#                     out_ch = self.out_channels[layer_idx]
#                 else:
#                     out_ch = self.skip_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
#                     in_ch = self.skip_channels[layer_idx - 1]
#                 blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
#         blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
#             self.in_channels[-1], 0, self.out_channels[-1], **kwargs
#         )
#         self.blocks = nn.ModuleDict(blocks)
#         self.depth = len(self.in_channels) - 1
#
#     def forward(self, *features):
#
#         features = features[1:]  # remove first skip with same spatial resolution
#         features = features[::-1]  # reverse channels to start from head of encoder
#         # start building dense connections
#         dense_x = {}
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(self.depth - layer_idx):
#                 if layer_idx == 0:
#                     output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
#                     dense_x[f"x_{depth_idx}_{depth_idx}"] = output
#                 else:
#                     dense_l_i = depth_idx + layer_idx
#                     cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
#                     cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
#                     dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
#                         dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
#                     )
#         dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
#         return dense_x[f"x_{0}_{self.depth}"]

class UnetPlusPlusDecoder_With_Edge(nn.Module): ## 기본 Unet plpl segment decoder
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        return dense_x[f"x_{0}_{self.depth}"]


# class UnetPlusPlusDecoder_With_Edge(nn.Module): ## X3,1 X2,2 X1,3 X0,4 에서 edge map 과 결합하는 segment decoder --> 결합 위치는 코드 내에서 수정 가능, 결합 한 후에는 conv 로 원래 채널과 동일하게 맞춤
#     def __init__(
#         self,
#         encoder_channels,
#         decoder_channels,
#         n_blocks=5,
#         use_batchnorm=True,
#         attention_type=None,
#         center=False,
#     ):
#         super().__init__()
#
#         if n_blocks != len(decoder_channels):
#             raise ValueError(
#                 "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
#                     n_blocks, len(decoder_channels)
#                 )
#             )
#
#         # remove first skip with same spatial resolution
#         encoder_channels = encoder_channels[1:]
#         # reverse channels to start from head of encoder
#         encoder_channels = encoder_channels[::-1]
#
#         # computing blocks input and output channels
#         head_channels = encoder_channels[0]
#         self.in_channels = [head_channels] + list(decoder_channels[:-1])
#         self.skip_channels = list(encoder_channels[1:]) + [0]
#         self.out_channels = decoder_channels
#         if center:
#             self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
#         else:
#             self.center = nn.Identity()
#
#         # combine decoder keyword arguments
#         kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
#
#         blocks = {}
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(layer_idx + 1):
#                 if depth_idx == 0:
#                     in_ch = self.in_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
#                     out_ch = self.out_channels[layer_idx]
#                 else:
#                     out_ch = self.skip_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
#                     in_ch = self.skip_channels[layer_idx - 1]
#                 blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
#         blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
#             self.in_channels[-1], 0, self.out_channels[-1], **kwargs
#         )
#
#         # 추가된 블록을 저장
#         # blocks[f"enhanced_x_3_1"] = DecoderBlock(self.out_channels[3], 0, self.out_channels[3], **kwargs)
#         # blocks[f"enhanced_x_2_2"] = DecoderBlock(self.out_channels[2], 0, self.out_channels[2], **kwargs)
#         blocks[f"enhanced_x_1_3"] = DecoderBlock(self.out_channels[1], 0, self.out_channels[1], **kwargs)
#         blocks[f"enhanced_x_0_4"] = DecoderBlock(self.out_channels[0], 0, self.out_channels[0], **kwargs)
#
#         self.blocks = nn.ModuleDict(blocks)
#         self.depth = len(self.in_channels) - 1
#
#     def forward(self, *features, edge_map):
#
#         features = features[1:]  # remove first skip with same spatial resolution
#         features = features[::-1]  # reverse channels to start from head of encoder
#         # start building dense connections
#         dense_x = {}
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(self.depth - layer_idx):
#                 if layer_idx == 0:
#                     output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
#                     dense_x[f"x_{depth_idx}_{depth_idx}"] = output
#                 else:
#                     dense_l_i = depth_idx + layer_idx
#                     cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
#                     cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
#                     dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
#                         dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
#                     )
#
#                 # X3,1 위치에서 edge_map을 결합하여 출력 생성
#                 # if layer_idx == 3 and depth_idx == 1:
#                 #     edge_map_resized = F.interpolate(edge_map, size=dense_x[f"x_{depth_idx}_{layer_idx}"].shape[2:], mode='bilinear', align_corners=False)
#                 #     enhanced_features = torch.cat([dense_x[f"x_{depth_idx}_{layer_idx}"], edge_map_resized], dim=1)
#                 #     # Conv 레이어로 채널 수 조정
#                 #     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=self.out_channels[3], kernel_size=1).to(enhanced_features.device)
#                 #     enhanced_features = conv(enhanced_features)
#                 #     dense_x[f"x_{depth_idx}_{layer_idx}"] = self.blocks[f"enhanced_x_3_1"](enhanced_features)
#                 #
#                 # # X2,2 위치에서 edge_map을 결합하여 출력 생성
#                 # if layer_idx == 2 and depth_idx == 2:
#                 #     edge_map_resized = F.interpolate(edge_map, size=dense_x[f"x_{depth_idx}_{layer_idx}"].shape[2:], mode='bilinear', align_corners=False)
#                 #     enhanced_features = torch.cat([dense_x[f"x_{depth_idx}_{layer_idx}"], edge_map_resized], dim=1)
#                 #     # Conv 레이어로 채널 수 조정
#                 #     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=self.out_channels[2], kernel_size=1).to(enhanced_features.device)
#                 #     enhanced_features = conv(enhanced_features)
#                 #     dense_x[f"x_{depth_idx}_{layer_idx}"] = self.blocks[f"enhanced_x_2_2"](enhanced_features)
#
#                 # X1,3 위치에서 edge_map을 결합하여 출력 생성
#                 if layer_idx == 1 and depth_idx == 3:
#                     edge_map_resized = F.interpolate(edge_map, size=dense_x[f"x_{depth_idx}_{layer_idx}"].shape[2:], mode='bilinear', align_corners=False)
#                     enhanced_features = torch.cat([dense_x[f"x_{depth_idx}_{layer_idx}"], edge_map_resized], dim=1)
#                     # Conv 레이어로 채널 수 조정
#                     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=self.out_channels[1], kernel_size=1).to(enhanced_features.device)
#                     enhanced_features = conv(enhanced_features)
#                     dense_x[f"x_{depth_idx}_{layer_idx}"] = self.blocks[f"enhanced_x_1_3"](enhanced_features)
#
#                 # X0,4 위치에서 edge_map을 결합하여 출력 생성
#                 if layer_idx == 0 and depth_idx == 4:
#                     edge_map_resized = F.interpolate(edge_map, size=dense_x[f"x_{depth_idx}_{layer_idx}"].shape[2:], mode='bilinear', align_corners=False)
#                     enhanced_features = torch.cat([dense_x[f"x_{depth_idx}_{layer_idx}"], edge_map_resized], dim=1)
#                     # Conv 레이어로 채널 수 조정
#                     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=self.out_channels[0], kernel_size=1).to(enhanced_features.device)
#                     enhanced_features = conv(enhanced_features)
#                     dense_x[f"x_{depth_idx}_{layer_idx}"] = self.blocks[f"enhanced_x_0_4"](enhanced_features)
#
#         dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
#         return dense_x[f"x_{0}_{self.depth}"]


# class UnetPlusPlusDecoder_With_Edge(nn.Module): ## X3,1 X2,2 X1,3 X0,4 에서 edge map 과 결합하는 segment decoder --> 결합 위치는 코드 내에서 수정 가능, 결합 한 후에는 conv 로 원래 채널과 동일하게 맞춤
#     def __init__(
#         self,
#         encoder_channels,
#         decoder_channels,
#         n_blocks=5,
#         use_batchnorm=True,
#         attention_type=None,
#         center=False,
#     ):
#         super().__init__()
#
#         if n_blocks != len(decoder_channels):
#             raise ValueError(
#                 "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
#                     n_blocks, len(decoder_channels)
#                 )
#             )
#
#         # remove first skip with same spatial resolution
#         encoder_channels = encoder_channels[1:]
#         # reverse channels to start from head of encoder
#         encoder_channels = encoder_channels[::-1]
#
#         # computing blocks input and output channels
#         head_channels = encoder_channels[0]
#         self.in_channels = [head_channels] + list(decoder_channels[:-1])
#         self.skip_channels = list(encoder_channels[1:]) + [0]
#         self.out_channels = decoder_channels
#         if center:
#             self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
#         else:
#             self.center = nn.Identity()
#
#         # combine decoder keyword arguments
#         kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
#
#         blocks = {}
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(layer_idx + 1):
#                 if depth_idx == 0:
#                     in_ch = self.in_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
#                     out_ch = self.out_channels[layer_idx]
#                 else:
#                     out_ch = self.skip_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
#                     in_ch = self.skip_channels[layer_idx - 1]
#                 blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
#         blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
#             self.in_channels[-1], 0, self.out_channels[-1], **kwargs
#         )
#
#         # 추가된 블록을 저장
#         # blocks[f"enhanced_x_3_1"] = DecoderBlock(self.out_channels[3], 0, self.out_channels[3], **kwargs)
#         # blocks[f"enhanced_x_2_2"] = DecoderBlock(self.out_channels[2], 0, self.out_channels[2], **kwargs)
#         blocks[f"enhanced_x_1_3"] = DecoderBlock(self.out_channels[1], 0, self.out_channels[1], **kwargs)
#         blocks[f"enhanced_x_0_4"] = DecoderBlock(self.out_channels[0], 0, self.out_channels[0], **kwargs)
#
#         self.blocks = nn.ModuleDict(blocks)
#         self.depth = len(self.in_channels) - 1
#
#     def forward(self, *features, edge_map):
#
#         features = features[1:]  # remove first skip with same spatial resolution
#         features = features[::-1]  # reverse channels to start from head of encoder
#         # start building dense connections
#         dense_x = {}
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(self.depth - layer_idx):
#                 if layer_idx == 0:
#                     output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
#                     dense_x[f"x_{depth_idx}_{depth_idx}"] = output
#                 else:
#                     dense_l_i = depth_idx + layer_idx
#                     cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
#                     cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
#                     dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
#                         dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
#                     )
#
#                 # X3,1 위치에서 edge_map을 결합하여 출력 생성
#                 # if layer_idx == 3 and depth_idx == 1:
#                 #     edge_map_resized = F.interpolate(edge_map, size=dense_x[f"x_{depth_idx}_{layer_idx}"].shape[2:], mode='bilinear', align_corners=False)
#                 #     enhanced_features = torch.cat([dense_x[f"x_{depth_idx}_{layer_idx}"], edge_map_resized], dim=1)
#                 #     # Conv 레이어로 채널 수 조정
#                 #     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=self.out_channels[3], kernel_size=1).to(enhanced_features.device)
#                 #     enhanced_features = conv(enhanced_features)
#                 #     dense_x[f"x_{depth_idx}_{layer_idx}"] = self.blocks[f"enhanced_x_3_1"](enhanced_features)
#                 #
#                 # # X2,2 위치에서 edge_map을 결합하여 출력 생성
#                 # if layer_idx == 2 and depth_idx == 2:
#                 #     edge_map_resized = F.interpolate(edge_map, size=dense_x[f"x_{depth_idx}_{layer_idx}"].shape[2:], mode='bilinear', align_corners=False)
#                 #     enhanced_features = torch.cat([dense_x[f"x_{depth_idx}_{layer_idx}"], edge_map_resized], dim=1)
#                 #     # Conv 레이어로 채널 수 조정
#                 #     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=self.out_channels[2], kernel_size=1).to(enhanced_features.device)
#                 #     enhanced_features = conv(enhanced_features)
#                 #     dense_x[f"x_{depth_idx}_{layer_idx}"] = self.blocks[f"enhanced_x_2_2"](enhanced_features)
#
#                 # X1,3 위치에서 edge_map을 결합하여 출력 생성
#                 if layer_idx == 1 and depth_idx == 3:
#                     edge_map_resized = F.interpolate(edge_map, size=dense_x[f"x_{depth_idx}_{layer_idx}"].shape[2:], mode='bilinear', align_corners=False)
#                     enhanced_features = torch.cat([dense_x[f"x_{depth_idx}_{layer_idx}"], edge_map_resized], dim=1)
#                     # Conv 레이어로 채널 수 조정
#                     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=self.out_channels[1], kernel_size=1).to(enhanced_features.device)
#                     enhanced_features = conv(enhanced_features)
#                     dense_x[f"x_{depth_idx}_{layer_idx}"] = self.blocks[f"enhanced_x_1_3"](enhanced_features)
#
#                 # X0,4 위치에서 edge_map을 결합하여 출력 생성
#                 if layer_idx == 0 and depth_idx == 4:
#                     edge_map_resized = F.interpolate(edge_map, size=dense_x[f"x_{depth_idx}_{layer_idx}"].shape[2:], mode='bilinear', align_corners=False)
#                     enhanced_features = torch.cat([dense_x[f"x_{depth_idx}_{layer_idx}"], edge_map_resized], dim=1)
#                     # Conv 레이어로 채널 수 조정
#                     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=self.out_channels[0], kernel_size=1).to(enhanced_features.device)
#                     enhanced_features = conv(enhanced_features)
#                     dense_x[f"x_{depth_idx}_{layer_idx}"] = self.blocks[f"enhanced_x_0_4"](enhanced_features)
#
#         dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
#         return dense_x[f"x_{0}_{self.depth}"]
