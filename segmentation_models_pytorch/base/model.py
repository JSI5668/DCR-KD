import torch
from . import initialization as init
import torch.nn.functional as F
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        device = x.device  # 입력 텐서가 위치한 장치(GPU 또는 CPU) 확인
        self.fc = self.fc.to(device)  # Channel Attention의 Conv2d 레이어를 입력 텐서와 동일한 장치로 이동

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2  # 동일한 출력 크기를 유지하기 위해 패딩 계산
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = x.device
        # Channel-wise 평균 및 최대 값 계산
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 두 가지 맵을 concat
        concat_out = torch.cat([avg_out, max_out], dim=1)
        self.conv = self.conv.to(device)
        self.sigmoid = self.sigmoid.to(device)
        # Conv 레이어 적용
        out = self.conv(concat_out)
        return self.sigmoid(out)


class AT(nn.Module):
    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        am = torch.div(am, norm+eps)

        return am

class SP(nn.Module):
    '''
	Similarity-Preserving Knowledge Distillation
	https://arxiv.org/pdf/1907.09682.pdf
	'''
    def __init__(self):
        super(SP, self).__init__()
    def forward(self, fm_s):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s  = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        # fm_t = fm_t.view(fm_t.size(0), -1)
		# G_t  = torch.mm(fm_t, fm_t.t())
		# norm_G_t = F.normalize(G_t, p=2, dim=1)

		# loss = F.mse_loss(norm_G_s, norm_G_t)

		# return loss
        return norm_G_s


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x): ## edge 추가
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x) ## edge 추가
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks, decoder_output ## --> DCRM 뽑을 때만 (feature 가 필요하니까, 평소에는 masks 만 return 받도록)

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


# class SegmentationModel_teacher(nn.Module):  ## Encoder 의 마지막 feature 와 edge decoder 의 마지막 feature 을 channel attention
#     def initialize(self):
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x):
#         """Sequentially pass `x` through model's encoder, edge decoder, and segment decoder"""
#
#         self.check_input_shape(x)
#
#         # 1. Encode input image
#         features = self.encoder(x)  # Shared encoder의 feature 추출
#         shared_encoder_last_feat = features[-1]  # Shared encoder의 마지막 feature
#
#         # 2. Decode edge features
#         edge_output, edge_feature_maps = self.decoder_edge(*features)  # Edge decoder의 마지막 feature 추출
#
#         # 3. Masks from edge decoder's last output
#         masks_edge = self.edge_head(edge_output)  # Edge mask 생성
#
#         # 4. Channel Attention 적용 (edge decoder의 마지막 feature에)
#         channel_attention = ChannelAttention(edge_feature_maps[-1].shape[1])  # Edge decoder의 마지막 feature map에 Attention 적용
#         edge_feat_last = edge_feature_maps[-1]
#         edge_feat_last = edge_feat_last.to(shared_encoder_last_feat.device)  # edge feature map도 동일한 장치로 이동
#         edge_feat_last_attended = channel_attention(edge_feat_last) * edge_feat_last  # 중요 채널 강조
#
#         edge_feat_resized_last = F.interpolate(edge_feat_last_attended, size=shared_encoder_last_feat.shape[2:], mode='bilinear', align_corners=False)
#
#         ##### 결합 방식 선택: Additive 또는 Concatenation #####
#
#         # Additive 방식으로 결합
#         fused_features_additive = shared_encoder_last_feat + edge_feat_resized_last
#
#         # # Concatenation 방식으로 결합
#         # combined_features_concat = torch.cat([shared_encoder_last_feat, edge_feat_resized_last], dim=1)
#         # conv_concat = nn.Conv2d(in_channels=combined_features_concat.shape[1], out_channels=shared_encoder_last_feat.shape[1], kernel_size=1).to(combined_features_concat.device)
#         # fused_features_concat = conv_concat(combined_features_concat)
#
#         # 6. Segment decoder 적용 (결합된 feature 사용)
#         decoder_output = self.decoder(fused_features_additive)  # 결합된 feature map을 segment decoder에 전달
#
#         # 7. Segment output 생성
#         masks = self.segmentation_head(decoder_output)  # 최종 segmentation 출력
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])  # 분류 작업이 필요할 경우
#             return masks, labels
#
#         return masks_edge, masks



# class SegmentationModel_teacher(torch.nn.Module):  ## edge decoder 의 초기 feature 을 channel attention 한 후, segment decoder 와 결합
#     def initialize(self):
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x):
#         """Pass `x` through encoder, apply Channel Attention to initial-layer of edge decoder, and combine with segment decoder"""
#
#         self.check_input_shape(x)
#
#         # Encode input image
#         features = self.encoder(x)
#
#         # Decode edge features, extracting edge_feature_maps
#         edge_output, edge_feature_maps = self.decoder_edge(*features)
#
#         # Masks from edge decoder's last output
#         masks_edge = self.edge_head(edge_output)  # 마지막 feature map에서 masks_edge 생성
#
#         # 초기 레이어에 Channel Attention 적용 (가정: edge_feature_maps[0]이 초기 레이어)
#         initial_edge_feature = edge_feature_maps[0]  # Edge Decoder의 초기 feature map
#         channel_attention = ChannelAttention(initial_edge_feature.shape[1])
#         initial_edge_feature_attended = channel_attention(initial_edge_feature) * initial_edge_feature
#
#         conv_concat = nn.Conv2d(initial_edge_feature_attended.shape[1], features[4].shape[1], kernel_size=1).to(initial_edge_feature_attended.device)
#         initial_edge_feature_attended = conv_concat(initial_edge_feature_attended)
#         features_attention_first_layer = features[4] + initial_edge_feature_attended
#         features[4] = features_attention_first_layer
#
#         # Segment Decoder를 통해 feature decoding
#         decoder_output = self.decoder(*features)
#
#         # Segment output
#         masks = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels
#
#         return masks_edge, masks


class SegmentationModel_teacher_forGradCAM(torch.nn.Module):  ## edge decoder 의 마지막 feature 을 channel attention 한 후, segment decoder 의 마지막과 결합
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` through model's encoder, edge decoder, and segment decoder"""

        self.check_input_shape(x)

        # Encode input image
        features = self.encoder(x)

        # Decode edge features
        edge_output, edge_feature_maps = self.decoder_edge(*features)

        # Masks from edge decoder's last output
        masks_edge = self.edge_head(edge_output)  # 마지막 feature map에서 masks_edge 생성


        ############### Ablation ###############################
        ############### Spatial Attention ###############
        # channel_attention = SpatialAttention(kernel_size=7)
        # edge_feat_last = edge_feature_maps[-1]
        # edge_feat_last_attended = channel_attention(edge_feat_last) * edge_feat_last  # 중요 채널 강조
        ##########################################################################################
        ############### just sum ###############
        # edge_feat_last = edge_feature_maps[-1]
        # edge_feat_last_attended = edge_feat_last
        ##########################################################################################
        ############### Proposed (Channel Attention) ###############
        # Channel Attention 적용
        channel_attention = ChannelAttention(edge_feature_maps[-1].shape[1])  # 마지막 edge feature map에 Attention 적용
        edge_feat_last = edge_feature_maps[-1]
        edge_feat_last_attended = channel_attention(edge_feat_last) * edge_feat_last  # 중요 채널 강조
        ##########################################################################################

        decoder_output = self.decoder(*features)

        ##### 결합 방식 선택: Additive 또는 Concatenation #####

        # Additive 방식으로 결합
        # fused_features_additive = decoder_output + (0.05*edge_feat_last_attended)
        fused_features_additive = decoder_output + edge_feat_last_attended

        # # Concatenation 방식으로 결합
        # combined_features_concat = torch.cat([decoder_output, edge_feat_last_attended], dim=1)
        # conv_concat = nn.Conv2d(in_channels=combined_features_concat.shape[1], out_channels=decoder_output.shape[1],
        #                         kernel_size=1).to(combined_features_concat.device)
        # fused_features_concat = conv_concat(combined_features_concat)


        # Segment output
        masks = self.segmentation_head(fused_features_additive)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        # return masks_edge, masks, fused_features_additive
        return masks

class SegmentationModel_teacher(torch.nn.Module):  ## edge decoder 의 마지막 feature 을 channel attention 한 후, segment decoder 의 마지막과 결합
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` through model's encoder, edge decoder, and segment decoder"""

        self.check_input_shape(x)

        # Encode input image
        features = self.encoder(x)

        # Decode edge features
        edge_output, edge_feature_maps = self.decoder_edge(*features)

        # Masks from edge decoder's last output
        masks_edge = self.edge_head(edge_output)  # 마지막 feature map에서 masks_edge 생성


        ############### Ablation ###############################
        ############### Spatial Attention ###############
        # channel_attention = SpatialAttention(kernel_size=7)
        # edge_feat_last = edge_feature_maps[-1]
        # edge_feat_last_attended = channel_attention(edge_feat_last) * edge_feat_last  # 중요 채널 강조
        ##########################################################################################
        ############### just sum ###############
        edge_feat_last = edge_feature_maps[-1]
        edge_feat_last_attended = edge_feat_last
        ##########################################################################################
        ############### Proposed (Channel Attention) ###############
        # Channel Attention 적용
        # channel_attention = ChannelAttention(edge_feature_maps[-1].shape[1])  # 마지막 edge feature map에 Attention 적용
        # edge_feat_last = edge_feature_maps[-1]
        # edge_feat_last_attended = channel_attention(edge_feat_last) * edge_feat_last  # 중요 채널 강조
        ##########################################################################################

        decoder_output = self.decoder(*features)

        ##### 결합 방식 선택: Additive 또는 Concatenation #####

        # Additive 방식으로 결합
        # fused_features_additive = decoder_output + (0.05*edge_feat_last_attended)
        fused_features_additive = decoder_output + edge_feat_last_attended

        # # Concatenation 방식으로 결합
        # combined_features_concat = torch.cat([decoder_output, edge_feat_last_attended], dim=1)
        # conv_concat = nn.Conv2d(in_channels=combined_features_concat.shape[1], out_channels=decoder_output.shape[1],
        #                         kernel_size=1).to(combined_features_concat.device)
        # fused_features_concat = conv_concat(combined_features_concat)


        # Segment output
        masks = self.segmentation_head(fused_features_additive)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks_edge, masks, fused_features_additive


# class SegmentationModel_teacher(torch.nn.Module):  ## KD (Spatial attention, SP) ## edge decoder 의 마지막 feature 을 channel attention 한 후, segment decoder 의 마지막과 결합
#     def initialize(self):
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x):
#         self.check_input_shape(x)
#
#         # Encode input image
#         features = self.encoder(x)
#
#         # Decode edge features
#         edge_output, edge_feature_maps = self.decoder_edge(*features)
#
#         # Masks from edge decoder's last output
#         masks_edge = self.edge_head(edge_output)
#
#         # Channel Attention 적용
#         channel_attention = ChannelAttention(edge_feature_maps[-1].shape[1])
#         edge_feat_last = edge_feature_maps[-1]
#         edge_feat_last_attended = channel_attention(edge_feat_last) * edge_feat_last
#
#         decoder_output = self.decoder(*features)
#
#         # Additive 방식으로 결합
#         fused_features_additive = decoder_output + edge_feat_last_attended
#
#         # # Spatial Attention 적용
#         # spatial_attention = SpatialAttention(kernel_size=7)
#         # spatial_attention_map_teacher = spatial_attention(fused_features_additive)
#
#         # AT 적용
#         # attention_map_teacher = AT(p=2)(fused_features_additive)
#
#         # SP 적용
#         sp = SP()
#         sp_teacher = sp(fused_features_additive)
#
#         # Segment output
#         masks = self.segmentation_head(fused_features_additive)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels, sp_teacher
#
#         return masks_edge, masks, sp_teacher

# class SegmentationModel(torch.nn.Module): ## shared encoder 의 마지막 feature map 과 concat 한 후, 원래 shared encoder의 마지막 feature map 과 크기 동일하게 한 후, segment decoder
#
#     def initialize(self):
#         init.initialize_decoder(self.decoder_edge)
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.edge_head)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x): ## edge 추가
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""
#
#         self.check_input_shape(x)
#
#         features = self.encoder(x)
#
#         # Edge map generation
#         edge_output = self.decoder_edge(*features)
#         masks_edge = self.edge_head(edge_output)
#
#         # masks_edge를 features[-1]의 크기에 맞게 다운샘플링
#         masks_edge_resized = F.interpolate(masks_edge, size=features[-1].shape[2:], mode='bilinear',
#                                            align_corners=False)
#         # features[-1]과 masks_edge 결합
#         enhanced_features = torch.cat([features[-1], masks_edge_resized], dim=1)
#
#         # Conv layer로 원래 features[-1]과 동일한 크기로 조정
#         conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=features[-1].shape[1], kernel_size=1).to(enhanced_features.device)
#         enhanced_features = conv(enhanced_features)
#
#         # 디코더에 넘길 features 리스트 업데이트
#         features[-1] = enhanced_features
#
#         # 디코더에 업데이트된 features를 전달
#         decoder_output = self.decoder(*features)
#
#         masks_seg = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks_seg, labels
#
#         return masks_edge, masks_seg
#
#     @torch.no_grad()
#     def predict(self, x):
#         """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
#
#         Args:
#             x: 4D torch tensor with shape (batch_size, channels, height, width)
#
#         Return:
#             prediction: 4D torch tensor with shape (batch_size, classes, height, width)
#
#         """
#         if self.training:
#             self.eval()
#
#         x = self.forward(x)
#
#         return x
#
#


# class SegmentationModel_teacher(torch.nn.Module):  ## 각 블록의 초기단계
#     def initialize(self):
#         init.initialize_decoder(self.decoder_edge)
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x):
#         """Sequentially pass `x` through model's encoder, edge decoder, segment decoder, and heads"""
#         self.check_input_shape(x)
#
#         # Shared Encoder
#         features = self.encoder(x)
#
#         # Edge Decoder
#         edge_outputs, edge_feature_maps = self.decoder_edge(*features)  # edge_feature_maps 사용
#
#         masks_edge = self.edge_head(edge_outputs)  # 마지막 feature map에서 masks_edge 생성
#
#         # Segment Decoder: Edge Decoder에서 얻은 feature map들과 결합
#         enhanced_features_list = []
#         selected_indices = [1, 2, 3]  # `x_{1}_{1}`, `x_{2}_{2}`, `x_{3}_{3}`에 해당하는 indices
#
#         for idx, seg_feat in enumerate(features):
#             if idx in selected_indices:
#                 edge_feat = edge_feature_maps[idx]  # Edge Decoder에서 가져온 해당 feature map
#                 edge_feat_resized = F.interpolate(edge_feat, size=seg_feat.shape[2:], mode='bilinear',
#                                                   align_corners=False)
#
#                 # 결합 후 Conv layer로 원래 크기로 조정
#                 enhanced_features = torch.cat([seg_feat, edge_feat_resized], dim=1)
#                 conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=seg_feat.shape[1],
#                                  kernel_size=1).to(enhanced_features.device)
#                 enhanced_features = conv(enhanced_features)
#
#                 enhanced_features_list.append(enhanced_features)
#             else:
#                 enhanced_features_list.append(seg_feat)
#
#         # Remaining features를 결합하여 업데이트된 features를 segment decoder에 전달
#         decoder_output = self.decoder(*enhanced_features_list)
#
#         # Segmentation Head
#         masks = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels
#
#         return masks_edge, masks
#
#     @torch.no_grad()
#     def predict(self, x):
#         """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`"""
#         if self.training:
#             self.eval()
#         x = self.forward(x)
#         return x


# class SegmentationModel_teacher(torch.nn.Module):  ## encoder 와 edge decoder 의 같은 해상도 feature 결합
#     def initialize(self):
#         init.initialize_decoder(self.decoder_edge)
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.edge_head)
#         init.initialize_head(self.segmentation_head)
#         init.initialize_conv1x1(self.conv1x1)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider padding your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x):
#         """Sequentially pass `x` through encoder, edge decoder, and segmentation decoder."""
#
#         # Step 1: Extract features from encoder
#         features = self.encoder(x)  # encoder에서 추출한 feature maps
#
#         # Step 2: Pass features through edge decoder
#         edge_output_feature, edge_feature_maps = self.decoder_edge(*features)
#         masks_edge = self.edge_head(edge_output_feature)
#
#         # Step 3: Combine edge and encoder feature maps with the same resolution
#         combined_feature_maps = []
#         for enc_idx, enc_feat in enumerate(features):
#             if enc_feat.shape[2:] == (224, 224):
#                 # 해상도가 224x224인 경우는 결합하지 않고 그대로 넘김
#                 combined_feature_maps.append(enc_feat)
#             else:
#                 matched = False
#                 # edge feature maps 중에서 enc_feat와 해상도가 같은 것 찾아서 concat
#                 for edge_idx, edge_feat in enumerate(edge_feature_maps):
#                     if enc_feat.shape[2:] == edge_feat.shape[2:]:
#                         # 해상도가 같으면 concatenate
#                         combined = torch.cat([enc_feat, edge_feat], dim=1)
#                         if f"x_{enc_idx}" in self.conv1x1:  # 1x1 convolution 적용할 수 있는 경우
#                             # 1x1 convolution으로 encoder feature의 채널로 복구
#                             combined = self.conv1x1[f"x_{enc_idx}"](combined)
#                         combined_feature_maps.append(combined)
#                         matched = True
#                         break  # Stop searching once a match is found
#
#                 # 해상도가 맞는 edge feature map을 못 찾은 경우, 그냥 encoder feature map을 그대로 사용
#                 if not matched:
#                     combined_feature_maps.append(enc_feat)
#
#         # Step 4: Pass combined features through the segmentation decoder
#         decoder_output = self.decoder(*combined_feature_maps)
#
#         # Step 5: Pass the final output through the segmentation head
#         masks = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels
#
#         return masks_edge, masks
#
#     @torch.no_grad()
#     def predict(self, x):
#         if self.training:
#             self.eval()
#
#         x = self.forward(x)
#
#         return x



# class SegmentationModel_teacher(torch.nn.Module):  ## selected_indices = [4, 5, 6] --> edge_outputs[-1][4]: x_{0}_{2}, edge_outputs[-1][5]: x_{1}_{2}, edge_outputs[-1][6]: x_{2}_{2} --> decoder 의 중간단계
#     def initialize(self):
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x):
#         """Sequentially pass `x` trough model`s encoder, edge decoder, and segment decoder"""
#
#         self.check_input_shape(x)
#
#         # Encode input image
#         features = self.encoder(x)
#
#         # Decode edge features
#         edge_output, edge_feature_maps = self.decoder_edge(*features)
#
#         masks_edge = self.edge_head(edge_output)  # 마지막 feature map에서 masks_edge 생성
#
#         # Segment decoder with edge feature maps integration
#         enhanced_features_list = []
#         selected_indices = [4, 5, 6]  # Use middle resolution features from edge decoder
#
#         # Loop over each selected feature
#         for seg_feat, idx in zip(features, selected_indices):
#             edge_feat = edge_feature_maps[idx]
#             # Resize edge feature map to match segment feature map
#             edge_feat_resized = F.interpolate(edge_feat, size=seg_feat.shape[2:], mode='bilinear', align_corners=False)
#
#             # Concatenate and adjust with Conv layer
#             enhanced_features = torch.cat([seg_feat, edge_feat_resized], dim=1)
#             conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=seg_feat.shape[1], kernel_size=1).to(
#                 enhanced_features.device)
#             enhanced_features = conv(enhanced_features)
#
#             # Instead of appending to the list, replace the corresponding feature
#             enhanced_features_list.append(enhanced_features)
#
#         # Fill the remaining features that were not enhanced
#         remaining_features = features[len(selected_indices):]
#
#         # Combine enhanced and remaining features
#         combined_features = enhanced_features_list + remaining_features
#
#         # Pass the combined features through the segment decoder
#         decoder_output = self.decoder(*combined_features)
#
#         # Segment output
#         masks = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels
#
#         return masks_edge, masks


# class SegmentationModel_teacher(torch.nn.Module): ## 모든 edge decoder block 을 segment decoder block 에 결합
#     def initialize(self):
#         init.initialize_decoder(self.decoder_edge)
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x):
#         """Sequentially pass `x` through model's encoder, edge decoder, segment decoder, and heads"""
#         self.check_input_shape(x)
#
#         # Shared Encoder
#         features = self.encoder(x)
#
#         # Edge Decoder
#         edge_outputs, edge_feature_maps = self.decoder_edge(*features)  # edge_feature_maps 사용
#
#         # Segment Decoder: 모든 블록의 각 단계에서 결합
#         enhanced_features_list = []
#
#         # Adjusted for-loop to handle all layers of segment decoder
#         for layer_idx, seg_feat in enumerate(features):
#             if layer_idx < len(edge_feature_maps):
#                 edge_feat = edge_feature_maps[layer_idx]  # Edge Decoder에서 가져온 해당 feature map
#                 edge_feat_resized = F.interpolate(edge_feat, size=seg_feat.shape[2:], mode='bilinear',
#                                                   align_corners=False)
#
#                 # 결합 후 Conv layer로 원래 크기로 조정
#                 enhanced_features = torch.cat([seg_feat, edge_feat_resized], dim=1)
#                 conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=seg_feat.shape[1],
#                                  kernel_size=1).to(enhanced_features.device)
#                 enhanced_features = conv(enhanced_features)
#
#                 enhanced_features_list.append(enhanced_features)
#             else:
#                 enhanced_features_list.append(seg_feat)
#
#         # 디코더에 업데이트된 features를 전달
#         decoder_output = self.decoder(*enhanced_features_list)
#
#         # Segmentation Head
#         masks = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels
#
#         return masks, edge_outputs
#
#     @torch.no_grad()
#     def predict(self, x):
#         """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`"""
#         if self.training:
#             self.eval()
#         x = self.forward(x)
#         return x




###################################  Share Encoder 와 결합  ########################################################################################

# class SegmentationModel_teacher(torch.nn.Module):  ## edge decoder 의 feature map 들과 shared encoder 의 feature map 들을 결합한 후, segment decoder
#     def initialize(self):
#         init.initialize_decoder(self.decoder_edge)
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.edge_head)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x):
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""
#
#         self.check_input_shape(x)
#
#         features = self.encoder(x)
#
#         # Edge map generation
#         edge_outputs = self.decoder_edge(*features)  # 각 레벨의 feature map들을 반환
#
#         masks_edge = self.edge_head(edge_outputs[0])  # 마지막 feature map에서 masks_edge 생성
#
#         # # Segment Decoder와 Edge Decoder의 feature map들을 결합 (모두 결합)
#         # enhanced_features_list = []
#         # for seg_feat, edge_feat in zip(features, edge_outputs[-1]):
#         #     # Edge Decoder의 feature map을 Segment Decoder의 feature map 크기에 맞게 조정
#         #     edge_feat_resized = F.interpolate(edge_feat, size=seg_feat.shape[2:], mode='bilinear', align_corners=False)
#         #
#         #     # 결합 후 Conv layer로 원래 크기로 조정
#         #     enhanced_features = torch.cat([seg_feat, edge_feat_resized], dim=1)
#         #     conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=seg_feat.shape[1], kernel_size=1).to(enhanced_features.device)
#         #     enhanced_features = conv(enhanced_features)
#         #
#         #     enhanced_features_list.append(enhanced_features)
#
#         # 선택된 low-resolution (deep layers) 및 high-resolution (shallow layers) feature maps
#         low_res_indices = [0, 1]  # 가장 의미론적으로 풍부한 저해상도 feature map들
#         high_res_indices = [9, 10]  # 해상도가 높고 세부적인 정보를 포함한 feature map들
#
#         selected_indices = low_res_indices + high_res_indices
#
#         # 기존 features 리스트를 복사하여 수정할 준비를 합니다.
#         updated_features = list(features)
#
#         # 선택된 feature maps만을 사용하여 결합
#         for seg_feat, idx in zip(features, selected_indices):
#             edge_feat = edge_outputs[-1][idx]
#             # Edge Decoder의 feature map을 Segment Decoder의 feature map 크기에 맞게 조정
#             edge_feat_resized = F.interpolate(edge_feat, size=seg_feat.shape[2:], mode='bilinear', align_corners=False)
#
#             # 결합 후 Conv layer로 원래 크기로 조정
#             enhanced_features = torch.cat([seg_feat, edge_feat_resized], dim=1)
#             conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=seg_feat.shape[1], kernel_size=1).to(
#                 enhanced_features.device)
#             enhanced_features = conv(enhanced_features)
#
#             # 기존의 features를 대체
#             updated_features[selected_indices.index(idx)] = enhanced_features
#
#         # 디코더에 업데이트된 features를 전달
#         decoder_output = self.decoder(*updated_features)
#
#         masks_seg = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(updated_features[-1])
#             return masks_seg, labels
#
#         return masks_edge, masks_seg
        # return masks_edge, masks_seg, decoder_output


# class SegmentationModel_teacher(torch.nn.Module): ## edge map 을 Encoder 마지막이랑 연결
#         def initialize(self):
#             init.initialize_decoder(self.decoder_edge)
#             init.initialize_decoder(self.decoder)
#             init.initialize_head(self.edge_head)
#             init.initialize_head(self.segmentation_head)
#             if self.classification_head is not None:
#                 init.initialize_head(self.classification_head)
#
#         def check_input_shape(self, x):
#
#             h, w = x.shape[-2:]
#             output_stride = self.encoder.output_stride
#             if h % output_stride != 0 or w % output_stride != 0:
#                 new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#                 new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#                 raise RuntimeError(
#                     f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                     f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#                 )
#
#         def forward(self, x): ## edge 추가
#             """Sequentially pass `x` trough model`s encoder, decoder and heads"""
#
#             self.check_input_shape(x)
#
#             features = self.encoder(x)
#
#             # Edge map generation
#             edge_output = self.decoder_edge(*features)
#             masks_edge = self.edge_head(edge_output)
#
#             # masks_edge를 features[-1]의 크기에 맞게 다운샘플링
#             masks_edge_resized = F.interpolate(masks_edge, size=features[-1].shape[2:], mode='bilinear',
#                                                align_corners=False)
#             # features[-1]과 masks_edge 결합
#             enhanced_features = torch.cat([features[-1], masks_edge_resized], dim=1)
#
#             # Conv layer로 원래 features[-1]과 동일한 크기로 조정
#             conv = nn.Conv2d(in_channels=enhanced_features.shape[1], out_channels=features[-1].shape[1], kernel_size=1).to(enhanced_features.device)
#             enhanced_features = conv(enhanced_features)
#
#             # 디코더에 넘길 features 리스트 업데이트
#             features[-1] = enhanced_features
#
#             # 디코더에 업데이트된 features를 전달
#             decoder_output = self.decoder(*features)
#
#             masks_seg = self.segmentation_head(decoder_output)
#
#             if self.classification_head is not None:
#                 labels = self.classification_head(features[-1])
#                 return masks_seg, labels
#
#             return masks_edge, masks_seg, decoder_output
#
#         @torch.no_grad()
#         def predict(self, x):
#             """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
#
#             Args:
#                 x: 4D torch tensor with shape (batch_size, channels, height, width)
#
#             Return:
#                 prediction: 4D torch tensor with shape (batch_size, classes, height, width)
#
#             """
#             if self.training:
#                 self.eval()
#
#             x = self.forward(x)
#
#             return x



# class SegmentationModel_student(torch.nn.Module):
#     def initialize(self):
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def check_input_shape(self, x):
#
#         h, w = x.shape[-2:]
#         output_stride = self.encoder.output_stride
#         if h % output_stride != 0 or w % output_stride != 0:
#             new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
#             new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
#             raise RuntimeError(
#                 f"Wrong input shape height={h}, width={w}. Expected image height and width "
#                 f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
#             )
#
#     def forward(self, x): ## edge 추가
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""
#
#         self.check_input_shape(x)
#
#         features = self.encoder(x) ## edge 추가
#         decoder_output = self.decoder(*features)
#
#         masks = self.segmentation_head(decoder_output)
#
#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels
#
#         return masks, decoder_output
#
#     @torch.no_grad()
#     def predict(self, x):
#         """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
#
#         Args:
#             x: 4D torch tensor with shape (batch_size, channels, height, width)
#
#         Return:
#             prediction: 4D torch tensor with shape (batch_size, classes, height, width)
#
#         """
#         if self.training:
#             self.eval()
#
#         x = self.forward(x)
#
#         return x


class SegmentationModel_student(torch.nn.Module): ## KD (spatial attention map)
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        self.check_input_shape(x)

        # Encode input image
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        # # Spatial Attention 적용
        # spatial_attention = SpatialAttention(kernel_size=7)
        # spatial_attention_map_student = spatial_attention(decoder_output)

        # AT 적용
        # attention_map_student = AT(p=2)(decoder_output)

        # SP 적용
        # sp = SP()
        # sp_student = sp(decoder_output)

        # Segment output
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            # return masks, labels, sp_student
            return masks, labels

        # return masks, decoder_output, sp_student
        return masks, decoder_output

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()

        x = self.forward(x)
        return x
