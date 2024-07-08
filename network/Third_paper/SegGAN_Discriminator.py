import torch
import torch.nn as nn
import torch.nn.functional as F

# 판별기 정의
class SegmentationDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(SegmentationDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, padding=0)
        )

        self.fc = nn.Linear(1 * 1 * 1, 1)

    def forward(self, image, seg_map):
        combined_input = torch.cat((image, seg_map), dim=1)
        return self.model(combined_input)


class SimpleSegmentationDiscriminator_Weak(nn.Module):
    def __init__(self, input_channels):
        super(SimpleSegmentationDiscriminator_Weak, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels * 2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, padding=0)
        )

        self.fc = nn.Linear(1 * 1 * 1, 1)

    def forward(self, image, seg_map):
        combined_input = torch.cat((image, seg_map), dim=1)
        return self.model(combined_input)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(MultiScaleDiscriminator, self).__init__()
        self.scale1 = self._make_layers(input_channels)
        self.scale2 = self._make_layers(input_channels)
        self.scale3 = self._make_layers(input_channels)

    def _make_layers(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, padding=0)
        )

    def forward(self, image, seg_map):
        combined_input = torch.cat((image, seg_map), dim=1)
        scale1_out = self.scale1(combined_input)
        scale2_out = self.scale2(F.avg_pool2d(combined_input, 2))
        scale3_out = self.scale3(F.avg_pool2d(combined_input, 4))
        return [scale1_out, scale2_out, scale3_out]