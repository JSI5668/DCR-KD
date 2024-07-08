from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import Camvid, Kitti_sample, Camvid_proposed, Camvid_proposed_each_0, Marin_sample, Dacon_segmentation, Camvid_sample_smallclasses
from torchvision import transforms as T
from metrics import StreamSegMetrics
from main import validate
from main import get_dataset
from network.unet import UNet_attnetion_encoder_concat

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import cv2
from network.unet import UNet
from torch import optim

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, default='D:/Dataset/Camvid/camvid_original_1',#required=True
                        help="path to a single image or image directory")
    # parser.add_argument("--dataset", type=str, default='camvid',
    #                     choices=['voc', 'cityscapes', 'camvid'], help='Name of training set')
    parser.add_argument("--dataset", type=str, default='camvid_smallclasses',
                        choices=['camvid_smallclasses', 'cityscapes, camvid', 'camvid_open', 'camvid_45', 'camvid_60', 'camvid_deblur45_v2', 'camvid_deblur60_v2', 'camvid_sample','camvid_proposed', 'kitti_sample', 'camvid_proposed_each_0', 'marin_sample', 'Dacon'], help='Name of dataset')
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default='E:/Second_paper/Result_Image/Segmentation/Camvid/Original_Firstfold_smallclasses_noattention',
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=256)
    # parser.add_argument("--crop_size", type=int, default=176)

    # parser.add_argument("--ckpt", default='D:/checkpoint/Segmentation/camvid/torch_deeplabv3plus/camvid_original/best_deeplabv3plus_resnet50_camvid_os16.pth', type=str,
    #                     help="resume from checkpoint")

    # parser.add_argument("--ckpt", default='D:/checkpoint/Segmentation/camvid/SecondPaper/binary_class/best_deeplabv3plus_resnet50_camvid_proposed_os16.pth', type=str,
    #                     help="resume from checkpoint")

    # parser.add_argument("--ckpt", default='D:/checkpoint/Segmentation/camvid/SecondPaper/binary_class/best_deeplabv3plus_resnet50_camvid_proposed_os16.pth', type=str,
    #                     help="resume from checkpoint")

    parser.add_argument("--ckpt", default='E:/Second_paper/Checkpoint/Camvid/Original_firstfold/main_8_double_en_de/only_smallclasses_notattention/best_deeplabv3plus_resnet50_camvid_smallclasses_os16.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt_pre", default='D:/Second_Paper/Checkpoint/Camvid/Segmentation/Original_using_binary/Subcom_Firstfold_UNet_BCE_Focal_alpha0.25_r2_lr0.001_500epoch_BlackWhite_/best_deeplabv3plus_resnet50_camvid_proposed_os16.pth', type=str,
                        help="restore from checkpoint")

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    # mean = [0.485, 0.456, 0.406],
    # std = [0.229, 0.224, 0.225])
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
    if  opts.dataset.lower() == 'camvid_open':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_45':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_deblur45_v2':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_60':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_deblur60_v2':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_sample':
        opts.num_classes = 12
        decode_fn = Camvid.decode_target
    elif opts.dataset.lower() == 'camvid_proposed':
        opts.num_classes = 2
        decode_fn = Camvid_proposed.decode_target
    elif opts.dataset.lower() == 'kitti_sample':
        opts.num_classes = 12
        decode_fn = Kitti_sample.decode_target
    elif opts.dataset.lower() == 'camvid_proposed_each_0':
        opts.num_classes = 4
        decode_fn = Camvid_proposed_each_0.decode_target
    elif opts.dataset.lower() == 'camvid_smallclasses':
        opts.num_classes = 9
        decode_fn = Camvid_sample_smallclasses.decode_target
    elif opts.dataset.lower() == 'Dacon':
        opts.num_classes = 2
        decode_fn = Dacon_segmentation.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.png'), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    ###################################################################
    model_pre = UNet(n_channels=3, n_classes=2, bilinear=True)

    ###################################################################

    # model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    model = UNet(n_channels=3, n_classes=9, bilinear=True)

    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)


    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)

        ##################################
        checkpoint_pre = torch.load(opts.ckpt_pre, map_location=torch.device('cpu'))
        model_pre.load_state_dict(checkpoint_pre["model_state"])
        model_pre = nn.DataParallel(model_pre)
        model_pre.to(device)
        ##################################
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)

            pred_pre = model_pre(img) # HW
            pred_pre = nn.Sigmoid()(pred_pre)
            img = pred_pre[:,1,:,:].unsqueeze(dim=1) * img
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW

            # 길이랑 하늘 제외 segmentation output
            # img_2 = Image.open(img_path).convert('RGB')
            # img_2 = np.array(img_2)
            #
            # 길이랑 하늘 제외 segmentation output
            # for i in range(len(pred)):
            #     for j in range(len(pred[i])):
            #         if pred[i][j] == 0 or pred[i][j] == 3:
            #             img_2[i][j][0] = 0
            #             img_2[i][j][1] = 0
            #             img_2[i][j][2] = 0
            #
            # # colorized_preds = decode_fn(pred).astype('uint8')
            # colorized_preds = Image.fromarray(img_2)

            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)

            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'.png'))



if __name__ == '__main__':
    main()
