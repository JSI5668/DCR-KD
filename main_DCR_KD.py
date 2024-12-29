import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import torch.nn.functional as F
import pandas as pd
from ptflops import get_model_complexity_info

import torchvision.transforms.functional as TF

from torch import optim

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from torch.utils import data
from datasets import  Camvid_sample
from utils import ext_transforms_labelEdge_labelSeg as et
# from utils import ext_transforms_original as et
from utils import DynamicAttentionWeights
from metrics import StreamSegMetrics
# from torchsummary import summary
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from torchsummaryX import summary

import segmentation_models_pytorch as smp


from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional
from utils.loss import Edge_PerceptualLoss, GANLoss
from torchvision.utils import save_image

#python -m visdom.server
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    # parser.add_argument("--data_root", type=str, default='./datasets/data',
    #                     help="path to Dataset")
    parser.add_argument("--data_root", type=str, default='/path/',
                        help="path to Dataset")  ##crop size 바꿔주기

    parser.add_argument("--dataset", type=str, default='RGB2Edge',
                        choices=['RGB2Edge', 'camvid_sample'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")

    parser.add_argument("--total_itrs", type=int, default=87000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=224) ##513

    parser.add_argument("--ckpt_teacher",default='/path/best.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt_student",default='/path/best.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt",default='/path/best.pth', type=str,
                        help="restore from checkpoint")

    parser.add_argument("--continue_training", action='store_true', default=True)

    parser.add_argument("--overlay",  default=True)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")

    parser.add_argument("--val_interval", type=int, default=87,
                        help="epoch interval for eval (default: 100)")

    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097', #13570
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    if opts.dataset == 'RGB2Edge':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            # et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            # et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        test_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = RGB2_labelEdge_labelSeg(root=opts.data_root, split='train', transform=train_transform)

        val_dst = RGB2_labelEdge_labelSeg(root=opts.data_root, split='val', transform=val_transform)

        test_dst = RGB2_labelEdge_labelSeg(root=opts.data_root, split='test', transform=test_transform)


    if opts.dataset == 'camvid_sample':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        test_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Camvid_sample(root=opts.data_root, split='train', transform=train_transform)

        val_dst = Camvid_sample(root=opts.data_root, split='val', transform=val_transform)

        test_dst = Camvid_sample(root=opts.data_root, split='test', transform=test_transform)

    return train_dst, val_dst, test_dst


def validate(opts, model, loader, device, criterion_L1):
    """Perform validation and return the validation loss and scores."""
    model.eval()
    val_loss = 0
    total_samples = 0
    with torch.no_grad():
        for i, (images, edges, _) in enumerate(loader):
            images = images.to(device)
            edges = edges.to(device)

            outputs = model(images)
            loss = criterion_L1(outputs, edges)
            val_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    val_loss /= total_samples
    return val_loss

def validate_seg(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.overlay:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, _, labels) in tqdm(enumerate(loader)):
        # for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)[0]
            # outputs = slide_inference(model, images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

def val_validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.overlay:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        interval_loss = 0
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            # optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples, interval_loss






import torch


def pairwise_similarity_loss(teacher_logits, student_logits, margin=1.0):
    """클래스 간 유사성을 고려한 pairwise 손실 함수"""

    # 교사 모델의 클래스별 확률 계산
    teacher_probs = F.softmax(teacher_logits, dim=1)  # Shape: (4, 12, 224, 224)
    student_probs = F.softmax(student_logits, dim=1)  # Shape: (4, 12, 224, 224)

    # 배치와 공간 차원을 하나로 합침 (B * H * W, C)의 형태로 변환
    teacher_probs_flat = teacher_probs.view(teacher_probs.size(0), teacher_probs.size(1), -1)  # Shape: (4, 12, 224*224)
    student_probs_flat = student_probs.view(student_probs.size(0), student_probs.size(1), -1)  # Shape: (4, 12, 224*224)

    # 클래스 간 유사도 행렬 계산 (C, C) for each batch
    teacher_similarity = torch.matmul(teacher_probs_flat, teacher_probs_flat.transpose(1, 2))  # Shape: (4, 12, 12)
    student_similarity = torch.matmul(student_probs_flat, student_probs_flat.transpose(1, 2))  # Shape: (4, 12, 12)

    # 유사성 행렬 간의 차이를 기반으로 손실 계산
    # similarity_loss = F.mse_loss(teacher_similarity, student_similarity)
    similarity_loss = torch.nn.L1Loss()(teacher_similarity, student_similarity)

    # 마진을 적용한 손실
    similarity_loss = torch.clamp(similarity_loss - margin, min=0.0)

    return similarity_loss


# Attention map 생성 함수
def get_attention_map(feature_map):
    """feature map으로부터 attention map 생성"""
    # feature_map shape: (B, C, H, W)
    attention_map = feature_map.mean(dim=1, keepdim=True)  # 채널 차원에 대해 평균 --> 채널별 정보를 합산해서 공간 (spatial) 차원의 중요도
    attention_map = F.relu(attention_map)  # 음수 값 제거 --> 중요도가 음수인 경우를 배제하기 위한 비선형 연산
    attention_map = attention_map / (attention_map.max() + 1e-8)  # 0~1로 정규화
    return attention_map  # Shape: (B, 1, H, W)

class get_attention_map_CBAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(get_attention_map_CBAM, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = x.device
        self.conv = self.conv.to(device)

        # Global Average Pooling and Global Max Pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # GAP
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # GMP

        # Concatenate along the channel dimension
        concat = torch.cat([avg_out, max_out], dim=1)

        # Convolution and Sigmoid activation
        attention = self.sigmoid(self.conv(concat))

        # return x * attention
        return attention


import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure all layers are on the same device as the input
        device = x.device
        self.fc = self.fc.to(device)

        # Global Average Pooling and Global Max Pooling
        avg_out = self.avg_pool(x)  # (B, C, 1, 1)
        max_out = self.max_pool(x)  # (B, C, 1, 1)

        # Pass through fully connected layers
        avg_out = self.fc(avg_out)  # (B, C, 1, 1)
        max_out = self.fc(max_out)  # (B, C, 1, 1)

        # Combine and apply sigmoid
        out = avg_out + max_out  # Element-wise sum
        return self.sigmoid(out)




# Attention map을 사용한 class similarity loss
def pairwise_similarity_loss_with_attention(teacher_logits, student_logits, teacher_attention_map,
                                            student_attention_map, margin=1.0):
    """Teacher와 Student 각각의 attention map을 사용하는 pairwise 손실 함수"""

    # Attention map을 각각 logit 크기로 맞추기 (B, 1, H, W -> B, C, H, W 크기로 확장)
    teacher_attention_map_resized = teacher_attention_map.expand_as(teacher_logits)  # (B, 12, 224, 224)
    student_attention_map_resized = student_attention_map.expand_as(student_logits)  # (B, 12, 224, 224)

    # 교사 모델과 학생 모델의 클래스별 확률 계산 (Softmax)
    teacher_probs = F.softmax(teacher_logits, dim=1)  # (B, C, H, W)
    student_probs = F.softmax(student_logits, dim=1)  # (B, C, H, W)

    # Attention map을 각각의 확률 분포에 곱하여 가중치 적용
    weighted_teacher_probs = teacher_probs * teacher_attention_map_resized  # (B, C, H, W)
    weighted_student_probs = student_probs * student_attention_map_resized  # (B, C, H, W)

    # 배치와 공간 차원을 하나로 합침 (B * H * W, C)의 형태로 변환
    teacher_probs_flat = weighted_teacher_probs.view(weighted_teacher_probs.size(0), weighted_teacher_probs.size(1),
                                                     -1)  # (B, C, H*W)
    student_probs_flat = weighted_student_probs.view(weighted_student_probs.size(0), weighted_student_probs.size(1),
                                                     -1)  # (B, C, H*W)

    # 클래스 간 유사성 행렬 계산 (C, C)
    teacher_similarity = torch.matmul(teacher_probs_flat, teacher_probs_flat.transpose(1, 2))  # (B, C, C)
    student_similarity = torch.matmul(student_probs_flat, student_probs_flat.transpose(1, 2))  # (B, C, C)

    # 유사성 행렬 간의 차이를 기반으로 손실 계산 (L1 손실)
    similarity_loss = torch.nn.L1Loss()(teacher_similarity, student_similarity)

    # 마진을 적용한 손실
    similarity_loss = torch.clamp(similarity_loss - margin, min=0.0)

    return similarity_loss


# Dynaminc Attention map을 사용한 class similarity loss
def pairwise_similarity_loss_with_dynamic_attention(teacher_logits, student_logits, teacher_attention_map,
                                                    student_attention_map, margin=1.0, dynamic_weights=None):
    """동적 가중치가 적용된 attention map을 사용하는 class similarity loss"""

    # 동적 가중치 적용 (optional)
    # if dynamic_weights is not None:
    #     teacher_attention_map, student_attention_map = dynamic_weights(teacher_attention_map, student_attention_map)

    # Attention map을 logit 크기로 맞추기 (B, 1, H, W -> B, C, H, W 크기로 확장)
    teacher_attention_map_resized = teacher_attention_map.expand_as(teacher_logits)  # (B, 12, 224, 224)
    student_attention_map_resized = student_attention_map.expand_as(student_logits)  # (B, 12, 224, 224)

    # 교사 모델과 학생 모델의 클래스별 확률 계산 (Softmax)
    teacher_probs = F.softmax(teacher_logits, dim=1)  # (B, C, H, W)
    student_probs = F.softmax(student_logits, dim=1)  # (B, C, H, W)

    # Attention map을 각각의 확률 분포에 곱하여 가중치 적용
    weighted_teacher_probs = teacher_probs * teacher_attention_map_resized  # (B, C, H, W)
    weighted_student_probs = student_probs * student_attention_map_resized  # (B, C, H, W)

    # 배치와 공간 차원을 하나로 합침 (B * H * W, C)의 형태로 변환
    teacher_probs_flat = weighted_teacher_probs.view(weighted_teacher_probs.size(0), weighted_teacher_probs.size(1),
                                                     -1)  # (B, C, H*W)
    student_probs_flat = weighted_student_probs.view(weighted_student_probs.size(0), weighted_student_probs.size(1),
                                                     -1)  # (B, C, H*W)

    # 클래스 간 유사성 행렬 계산 (C, C)
    teacher_similarity = torch.matmul(teacher_probs_flat, teacher_probs_flat.transpose(1, 2))  # (B, C, C)
    student_similarity = torch.matmul(student_probs_flat, student_probs_flat.transpose(1, 2))  # (B, C, C)

    # 유사성 행렬 간의 차이를 기반으로 손실 계산 (L1 손실)
    similarity_loss = torch.nn.L1Loss()(teacher_similarity, student_similarity)

    # 마진을 적용한 손실
    similarity_loss = torch.clamp(similarity_loss - margin, min=0.0)

    return similarity_loss




def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'camvid':
        opts.num_classes = 12
    elif opts.dataset.lower() == 'camvid_sample':
        opts.num_classes = 12
    elif opts.dataset.lower() == 'kitti_sample':
        opts.num_classes = 12
    elif opts.dataset.lower() == 'mini_city':
        opts.num_classes = 20



    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst, test_dst = get_dataset(opts)


    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)

    test_loader = data.DataLoader(
        test_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)


    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(train_dst), len(val_dst), len(test_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    teacher_model = smp.UnetPlusPlus_teacher(encoder_name="resnext101_32x8d", encoder_weights="imagenet", in_channels=3, classes_edge=1, classes_seg=12)
    checkpoint_generator = torch.load(opts.ckpt_teacher, map_location=torch.device('cpu'))
    teacher_model.load_state_dict(checkpoint_generator["model_state"])
    teacher_model.cuda()

    student_model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=12)
    student_model.cuda()

    # optimizer_G_edge = optim.Adam(generator.parameters(), lr=opts.lr)
    optimizer = optim.RMSprop(student_model.parameters(), lr=opts.lr, weight_decay=1e-8, momentum=0.9)

    # Set up metrics
    metrics = StreamSegMetrics(12)

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)

    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion_seg = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        criterion_L1 = torch.nn.L1Loss()
        criterion_MSE = torch.nn.MSELoss()


    def save_ckpt(path):
        """ save current model
        """
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save({
            "cur_itrs": cur_itrs,
            # "model_state": model.module.state_dict(),  ##Data.parraell 이 있으면 module 이 생긴다.
            "model_state": student_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    def save_ckpt_with_dynamic(path):
        """ save current model
        """
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save({
            "cur_itrs": cur_itrs,
            # "model_state": model.module.state_dict(),  ##Data.parraell 이 있으면 module 이 생긴다.
            "model_state": student_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "dynamic_attention_state": dynamic_attention.state_dict()
        }, path)
        print("Model saved as %s" % path)


    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    total_train_miou = []
    total_train_loss_class_similarity = []
    total_train_loss_feature = []
    total_train_loss_seg = []
    total_val_miou = []
    total_val_loss_G = []

    total_train_loss_D = []
    total_val_loss_D = []

    if opts.ckpt_student is not None and opts.test_only:
        # pass
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan

        checkpoint_generator = torch.load(opts.ckpt_student, map_location=torch.device('cpu'))
        student_model.load_state_dict(checkpoint_generator["model_state"])

        student_model = nn.DataParallel(student_model)
        student_model.to(device)

        print("Model restored from %s" % opts.ckpt_student)
        # del checkpoint  # free memory
    else:
        # pass
        print("[!] Retrain")
        # model = nn.DataParallel(model)
        student_model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        # output_dir = 'F:/Third_paper/Codes/pytorch_deeplab_third_paper/outputs/RGB2Edge_Use_Unetplpl_noDis/CamVid_Firstfoldtest_Use_Firstfoldweight'
        # test_and_save_results(generator, test_loader, output_dir, device)

        student_model.eval()
        val_score, ret_samples = validate_seg(
            opts=opts, model=student_model, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))


        return

    else:
        interval_loss_class_similarity = 0
        interval_loss_feature = 0
        interval_loss_seg = 0

        interval_loss_class_similarity_plt = 0
        interval_loss_feature_plt = 0
        interval_loss_seg_plt = 0


        # Training loop
        dynamic_attention = DynamicAttentionWeights(num_channels_in=16, num_channels_out=12)  # 동적 가중치 모듈 초기화
        while True:  # cur_itrs < opts.total_itrs:


            student_model.train()
            cur_epochs += 1

            for (images, labels_edge, labels_seg) in train_loader:
                cur_itrs += 1
                images = images.to(device, dtype=torch.float32)
                labels_seg = labels_seg.to(device, dtype=torch.long)

                # Student model forward
                optimizer.zero_grad()
                # seg_outputs_student, feature_outputs_student, feature_outputs_student_spatial_attention_map = student_model(images)
                seg_outputs_student, feature_outputs_student = student_model(images)

                # Teacher model outputs for feature loss
                with torch.no_grad():
                    _,seg_outputs_teacher,feature_outputs_teacher = teacher_model(images)

                # Feature loss 계산
                feature_loss = criterion_L1(feature_outputs_student, feature_outputs_teacher)

                # 원래 task의 loss 계산 (예: segmentation loss)
                seg_loss = criterion_seg(seg_outputs_student, labels_seg)


                # Channel Attention map 생성 (decoder에서 추출한 feature map으로) --> CBAM 방식으로
                attention_map_student = ChannelAttention(in_planes=16, ratio=4)(feature_outputs_student)  ## --> 4, 16, 224, 224
                attention_map_teacher = ChannelAttention(in_planes=16, ratio=4)(feature_outputs_teacher)  # 교사 모델에서 추출

                # # 동적 가중치 적용
                weighted_attention_map_teacher, weighted_attention_map_student = dynamic_attention(
                    attention_map_teacher, attention_map_student)

                # 클래스간 유사성을 고려한 pairwise loss 계산 (각각의 attention map 및 동적 가중치 적용)
                class_similarity_loss = pairwise_similarity_loss_with_dynamic_attention(
                    seg_outputs_teacher, seg_outputs_student,
                    weighted_attention_map_teacher, weighted_attention_map_student
                )



                # 최종 loss 계산
                loss = seg_loss + feature_loss + 0.001 * class_similarity_loss

                # Backpropagation 및 optimization step
                loss.backward()
                optimizer.step()

                # ----- 손실 기록 및 출력 -----
                np_loss_class_similarity = class_similarity_loss.detach().cpu().numpy()
                np_loss_feature = feature_loss.detach().cpu().numpy()
                np_loss_seg = seg_loss.detach().cpu().numpy()

                interval_loss_class_similarity += np_loss_class_similarity
                interval_loss_feature += np_loss_feature
                interval_loss_seg += np_loss_seg

                interval_loss_class_similarity_plt += np_loss_class_similarity
                interval_loss_feature_plt += np_loss_feature
                interval_loss_seg_plt += np_loss_seg

                if vis is not None:
                    vis.vis_scalar('Loss_class_similarity', cur_itrs, np_loss_class_similarity)
                    vis.vis_scalar('Loss_feature', cur_itrs, np_loss_feature)
                    vis.vis_scalar('Loss_seg', cur_itrs, np_loss_seg)

                if (cur_itrs) % 10 == 0:
                    interval_loss_class_similarity /= 10
                    interval_loss_feature /= 10
                    interval_loss_seg /= 10
                    print(
                        f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss_class_similarity = {interval_loss_class_similarity}, Loss_feature={interval_loss_feature}, Loss_seg={interval_loss_seg}")

                    interval_loss_class_similarity = 0.0
                    interval_loss_feature = 0.0
                    interval_loss_seg = 0.0 ## 이 부분 오류 --> interval_loss_seg 이걸로 되야함

                ## train loss
                if (cur_itrs) % opts.val_interval == 0:
                    interval_loss_class_similarity_plt = interval_loss_class_similarity_plt / opts.val_interval
                    print("---------Epoch %d, Itrs %d/%d, train_Loss_feature=%f----------" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss_class_similarity_plt))
                    total_train_loss_class_similarity.append(interval_loss_class_similarity_plt)

                if (cur_itrs) % opts.val_interval == 0:
                    interval_loss_feature_plt = interval_loss_feature_plt / opts.val_interval
                    print("---------Epoch %d, Itrs %d/%d, train_Loss_feature=%f----------" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss_feature_plt))
                    total_train_loss_feature.append(interval_loss_feature_plt)

                if (cur_itrs) % opts.val_interval == 0:
                    interval_loss_seg_plt = interval_loss_seg_plt / opts.val_interval
                    print("---------Epoch %d, Itrs %d/%d, train_Loss_seg=%f----------" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss_seg_plt))
                    total_train_loss_seg.append(interval_loss_seg_plt)

                ## train miou
                if (cur_itrs) % opts.val_interval == 0:
                    train_val_score, ret_samples = validate_seg(
                        opts=opts, model=student_model, loader=train_loader, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id)
                    print(train_val_score['Mean IoU'])
                    print("---------Epoch %d, Itrs %d/%d, train_Miou=%f----------" %
                          (cur_epochs, cur_itrs, opts.total_itrs, train_val_score['Mean IoU']))

                    total_train_miou.append(train_val_score['Mean IoU'])

                if (cur_itrs) % opts.val_interval == 0:
                    # save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                    #           (opts.model, opts.dataset, opts.output_stride))
                    save_ckpt('/path/latest_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
                    print("validation...")
                    student_model.eval()
                    val_score, ret_samples = validate_seg(
                        opts=opts, model=student_model, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                    print(metrics.to_str(val_score))

                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        # save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                        #           (opts.model, opts.dataset,opts.output_stride))

                        save_ckpt_with_dynamic('/path/best_%s_%s_os%d.pth' %
                                  (opts.model, opts.dataset, opts.output_stride))

                    if vis is not None:  # visualize validation score and samples
                        vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                        for k, (img, target, lbl) in enumerate(ret_samples):
                            img = (denorm(img) * 255).astype(np.uint8)
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                            concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                            vis.vis_image('Sample %d' % k, concat_img)

                    save_ckpt_with_dynamic('/path/_%s_%s_%s_os%d.pth' %
                              (cur_epochs, opts.model, opts.dataset, opts.output_stride))
                    student_model.train()
                scheduler.step()

                if cur_itrs >= opts.total_itrs:
                    df_train_loss = pd.DataFrame(total_train_loss_seg)
                    df_train_miou = pd.DataFrame(total_train_miou)


                    plt.rcParams['axes.xmargin'] = 0
                    plt.rcParams['axes.ymargin'] = 0
                    plt.plot(total_train_miou)
                    plt.xlabel('epoch')
                    plt.ylabel('miou')
                    plt.show()

                    plt.plot(total_train_loss_seg)
                    plt.xlabel('epoch')
                    plt.ylabel('loss')
                    plt.show()

                    return


if __name__ == '__main__':
    main()