import os
import glob
import numpy as np
import PIL.Image
from PIL import Image
import re
import cv2
import torch
from typing import Optional
from torchvision.transforms import ToTensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = 'cuda',
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        ignore_index=255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1],
                [2, 0]]
            ])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],

                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],

                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype)
    # labels.cuda()
    # labels : (B, H, W)
    # labels.unsqueeze(1) #: (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.cuda().unsqueeze(1), 1.0) + eps

    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret



# path = 'E:/Second_paper/Figure/0016E_05970_blurred'
path = 'D:/Dataset/KITTI/MPRNet/psf_30_kitti_firstfold/test/input'

# files = glob.glob('C:\Users\JSIISPR\Desktop\github_deeplab/amazing/Amazing-Semantic-Segmentation-master/camvid_blurred45_twofold_gray/train/images/*.png')


imgNames = os.listdir(path)

for name in imgNames:
    try:
        # name -> 이미지 이름
        img_path = f'{path}/{name}'
        img = Image.open(img_path)
        # img_resize = img.resize((192, 192), resample=PIL.Image.NEAREST)
        img_resize = img.resize((512, 192), resample=PIL.Image.BILINEAR)

        img_resize.save(f'D:/Dataset/KITTI/segmentation/firstfold/psf_30_firstfold_192512/leftImg8bit/test/images/{name}')

    except:
        pass



# for name in imgNames:
#     # try:
#     # name -> 이미지 이름
#     img_path = f'{path}/{name}'
#     ##################################################################################################
#     # img = Image.open(img_path)
#     # img = ToTensor()(img)
#     # img = img * 255
#     # img_onehot = np.array(img.cpu())
#     # output = torch.zeros((240,320))
#     #################################################################################
#     img_color = cv2.imread(img_path)
#     img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#
#     img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
#     img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
#
#     img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
#     img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
#
#     img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
#     # img_sobel = img_sobel * 3
#     img_sobel = img_sobel
#     #################################################################################################
#     # img = Image.open(img_path)
#     # img = ToTensor()(img)
#     #
#     # img_onehot = label_to_one_hot_label(img.long(), num_classes=11, ignore_index=255)
#     # img_onehot = img_onehot * 255
#     # img_onehot = np.array(img_onehot.cpu())
#     # output = torch.zeros((240,320))
#     # for i in range(11):
#     #
#     #     img_sobel_x = cv2.Sobel(img_onehot[0][i], cv2.CV_64F, 1, 0, ksize=3)
#     #     img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
#     #
#     #     img_sobel_y = cv2.Sobel(img_onehot[0][i], cv2.CV_64F, 0, 1, ksize=3)
#     #     img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
#     #
#     #     img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
#     #
#     #     output += img_sobel
#     # # output = output * 10
#
#     # cv2.imwrite(f'E:/Second_paper/Data/Kitti/Edge_Sobel/train_mul5/{name}', img_sobel.squeeze(0))
#     cv2.imwrite(f'F:/Third_paper/Result_Image/Edge_mask/Sobel/RGB_Input/result/{name}', img_sobel)
# #     # cv2.imwrite(f'E:/Second_paper/Result_Image/Edge/test_onehot/{name}', output.numpy())



# for i in range(350):
    # for f in files:
    #     # name = os.listdir(path)
    #     img = Image.open(f)
    #     img_resize = img.resize((320, 240))
    #     title, ext = os.path.splitext(f)
    #
    #     name = f.split('/', aixs=-1)
    #     img_resize.save(
    #         'C:/Users/JSIISPR/Desktop/github_deeplab/Deblur_dataset/fold_A/fold_A_1/{name}'.format(name=name[i]))
    #     print(name[i])
    #     a = name[i]


# path = 'E:/Second_paper/Data/Mini_city/For_EdgeNet/Firstfold_mul_5/gtFine/train/images/aachen_000030_000019.png'
#
# img = Image.open(path)
#
# img_np = np.array(img)
