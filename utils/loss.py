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