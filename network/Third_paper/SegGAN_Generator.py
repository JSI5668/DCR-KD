import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Net 기반 Generator 정의
class UNetGenerator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNetGenerator, self).__init__()
        self.enc1 = self.conv_block(input_channels, 64)  # [B, 64, 224, 224]
        self.enc2 = self.conv_block(64, 128)            # [B, 128, 112, 112]
        self.enc3 = self.conv_block(128, 256)           # [B, 256, 56, 56]
        self.enc4 = self.conv_block(256, 512)           # [B, 512, 28, 28]
        self.bottleneck = self.conv_block(512, 1024)    # [B, 1024, 14, 14]
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # [B, 512, 28, 28]
        self.dec4 = self.conv_block(1024, 512)          # [B, 512, 28, 28]
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # [B, 256, 56, 56]
        self.dec3 = self.conv_block(512, 256)           # [B, 256, 56, 56]
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # [B, 128, 112, 112]
        self.dec2 = self.conv_block(256, 128)           # [B, 128, 112, 112]
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # [B, 64, 224, 224]
        self.dec1 = self.conv_block(128, 64)            # [B, 64, 224, 224]
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)  # [B, output_channels, 224, 224]

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 인코더
        enc1 = self.enc1(x)  # [B, 64, 224, 224]
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # [B, 128, 112, 112]
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # [B, 256, 56, 56]
        enc4 = self.enc4(F.max_pool2d(enc3, 2))  # [B, 512, 28, 28]
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))  # [B, 1024, 14, 14]

        # 디코더
        upconv4 = self.upconv4(bottleneck)  # [B, 512, 28, 28]
        dec4 = self.dec4(torch.cat((upconv4, enc4), dim=1))  # [B, 1024, 28, 28] -> [B, 512, 28, 28]
        upconv3 = self.upconv3(dec4)  # [B, 256, 56, 56]
        dec3 = self.dec3(torch.cat((upconv3, enc3), dim=1))  # [B, 512, 56, 56] -> [B, 256, 56, 56]
        upconv2 = self.upconv2(dec3)  # [B, 128, 112, 112]
        dec2 = self.dec2(torch.cat((upconv2, enc2), dim=1))  # [B, 256, 112, 112] -> [B, 128, 112, 112]
        upconv1 = self.upconv1(dec2)  # [B, 64, 224, 224]
        dec1 = self.dec1(torch.cat((upconv1, enc1), dim=1))  # [B, 128, 224, 224] -> [B, 64, 224, 224]

        return self.final_conv(dec1)  # [B, output_channels, 224, 224]
