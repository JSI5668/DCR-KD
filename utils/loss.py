import torch.nn as nn
import torch.nn.functional as F
import torch 
import timm

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

# PATH_1 = 'E:/Second_paper/Checkpoint/Camvid/EdgeNet_2_model/model.pt'
# model = torch.load(PATH_1)
#
# train_nodes, eval_nodes = get_graph_node_names(model)
#
# return_nodes={
#     train_nodes[2]: 'f1',
#     train_nodes[5]: 'f2',
#     train_nodes[7]: 'f3'
# }
#
# aa = create_feature_extractor(model,return_nodes)
# ooo = aa(inputs)
# ooo['f1']

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class Edge_PerceptualLoss(nn.Module):
    def __init__(self, model, feature_extract):
        super(Edge_PerceptualLoss, self).__init__()

        self.Perceptual_loss = nn.L1Loss()
        self.model = model
        self.feature_extract = feature_extract

    def forward(self, inputs, targets):
        with torch.no_grad():
            segment_output_edge_feature_1 = self.feature_extract(inputs)['f1']
            ground_truth_edge_feature_1 = self.feature_extract(targets)['f1']
            loss_1 = self.Perceptual_loss(segment_output_edge_feature_1, ground_truth_edge_feature_1)

            segment_output_edge_feature_2 = self.feature_extract(inputs)['f2']
            ground_truth_edge_feature_2 = self.feature_extract(targets)['f2']
            loss_2 = self.Perceptual_loss(segment_output_edge_feature_2, ground_truth_edge_feature_2)

            segment_output_edge_feature_3 = self.feature_extract(inputs)['f3']
            ground_truth_edge_feature_3 = self.feature_extract(targets)['f3']
            loss_3 = self.Perceptual_loss(segment_output_edge_feature_3, ground_truth_edge_feature_3)

            perceptual_loss_total = loss_1 + 0.5 * loss_2 + 0.25 * loss_3
            #
            # perceptual_loss_total = loss_1

        return perceptual_loss_total

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        ## flatten label and prediction tensors
        inputs = inputs.view(-1)
        # targets = targets.view(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE