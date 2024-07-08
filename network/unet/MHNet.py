import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch_utils import LayerNorm2d
from einops import rearrange
# from .unet_parts import Conv1x1, DoubleConv, Down, Up, OutConv, SCM, BasicConv, ChannelAttention
# from .unet_parts import *

###############################################################################################################
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.output(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



#######################################################################################################################
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv_my(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv_my(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv_my(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv_my(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv_my(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class BasicConv_my(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv_my, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ChannelAttention(nn.Module):
    def __init__(self, inplanes, ratio=16):
        super().__init__()
        planes = inplanes // ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, inplanes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
###############################################################################################################

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class Depth_wise(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=False, padding = 1, stride = 1):
        super(Depth_wise, self).__init__()
        self.depth_conv = nn.Conv2d(in_ch,
                                    in_ch,
                                    kernel_size,
                                    stride,
                                    padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size=1,
                                    padding=0, stride=1,
                                    groups=1)
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
#
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

#########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


#Dual Attention Unit
class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.ReLU(), res_scale=1):
        super(DAU, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat//2, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

class Sca_layer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(Sca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, stride=1,
                    bias=True),
        )


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(SCAB, self).__init__()
        modules_body = []
        modules_body.append(Depth_wise(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(Depth_wise(n_feat//2, n_feat, kernel_size, bias=bias))

        self.SCA = Sca_layer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.SCA(res)
        res += x
        return res


class UpDSample(nn.Module):
    def __init__(self, in_channels, n_feat):
        super(UpDSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                            nn.Conv2d(in_channels, n_feat, 1, stride=1, padding=0, bias=False)
                                )

    def forward(self, x):
        x = self.up(x)
        return x




class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class BotBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class Bottneck(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[BotBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[Attention(chan, 8, False) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[BotBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []


        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        decs = []
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            decs.append(x)

        x = self.ending(x)
        x = x + inp

        return x, encs, decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# Original Resolution Block (CABG)
class CABG(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(CABG, self).__init__()
        modules_body = []
        modules_body = [SCAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class CASNet(nn.Module):
    def __init__(self, n_feat,width, kernel_size, reduction, act, bias, num_cab):
        super(CASNet, self).__init__()

        self.info = nn.Conv2d(width,n_feat,kernel_size=1, bias=bias)

        self.CABG1 =  CABG(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.CABG2 = CABG(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.CABG3 = CABG(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.CABG4 = CABG(n_feat, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpDSample(width*2, n_feat)
        self.up_dec1 = UpDSample(width*2,n_feat)

        self.up_enc2 = nn.Sequential(UpDSample(width*4,width*2), UpDSample(width*2,n_feat))
        self.up_dec2 = nn.Sequential(UpDSample(width*4,width*2), UpDSample(width*2,n_feat))

        self.up_enc3 = nn.Sequential(UpDSample(width * 8,width*4), UpDSample(width*4,width*2), UpDSample(width*2,n_feat))
        self.up_dec3 = nn.Sequential(UpDSample(width * 8,width*4), UpDSample(width*4,width*2), UpDSample(width*2,n_feat))

        self.conv_enc1 = nn.Conv2d(width, n_feat, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_enc4 = nn.Conv2d(n_feat, n_feat , kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(width, n_feat, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_dec4 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm2d(n_feat)
        self.norm2 = LayerNorm2d(n_feat)
        self.norm3 = LayerNorm2d(n_feat)
        self.norm4 = LayerNorm2d(n_feat)

    def forward(self, x, encoder_outs, decoder_outs):
        x= self.info(x)
        x = self.norm1(x)
        x = self.CABG1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[3])

        x = self.norm2(x)
        x = self.CABG2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[2]))

        x = self.norm3(x)
        x = self.CABG3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[1]))

        x = self.norm4(x)
        x = self.CABG4(x)
        x = x + self.conv_enc4(self.up_enc3(encoder_outs[3])) + self.conv_dec4(self.up_dec3(decoder_outs[0]))

        return x

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class MHNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, width=32, n_feat=64, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super().__init__()
        act = SimpleGate()
        self.shallow_feat1 = nn.Sequential(conv(in_c, width, kernel_size, bias=bias),
                                           DAU(width, kernel_size, reduction, bias=bias, act=act))
        self.info = nn.Conv2d(width, n_feat, 1)
        self.stage1 = Bottneck(in_c, width)
        self.stage2 = CASNet(n_feat, width, kernel_size, reduction, act, bias,
                                    num_cab)
        self.concat12 = conv(width*2, width, kernel_size, bias=bias)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)
    
    def forward(self, x3_img):

        x1 = self.shallow_feat1(x3_img)

        x_s1, x1_en, x1_dn = self.stage1(x3_img)

        x2_cat = self.concat12(torch.cat([x1, x1_dn[3]], 1))
        x2_cat = self.stage2(x2_cat, x1_en, x1_dn)

        stage2_img = self.tail(x2_cat)

        return [stage2_img + x3_img, x_s1]


class MHNet_my(nn.Module):
    def __init__(self, in_c=3, out_c=3, width=32, n_feat=64, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super().__init__()
        act = SimpleGate()
        self.shallow_feat1 = nn.Sequential(conv(in_c, width, kernel_size, bias=bias),
                                           DAU(width, kernel_size, reduction, bias=bias, act=act))
        self.info = nn.Conv2d(width, n_feat, 1)
        self.stage1 = Bottneck(in_c, width)
        # self.stage2 = CASNet(n_feat, width, kernel_size, reduction, act, bias,
        #                      num_cab)
        self.concat12 = conv(width * 2, width, kernel_size, bias=bias)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)

        self.conv2 = conv(32, 3, kernel_size=1)
        self.stage2 = UNet_my_2(n_channels=3, n_classes=12, bilinear=True)

    def forward(self, x3_img):
        x1 = self.shallow_feat1(x3_img)

        x_s1, x1_en, x1_dn = self.stage1(x3_img)

        x2_cat = self.concat12(torch.cat([x1, x1_dn[3]], 1))
        x2_cat = self.conv2(x2_cat)
        x2_cat = x2_cat + x3_img
        x2_cat = self.stage2(x2_cat)

        # stage2_img = self.tail(x2_cat)

        return x2_cat

class MHNet_my_2(nn.Module):
    def __init__(self, in_c=3, out_c=3, width=32, n_feat=64, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super().__init__()
        act = SimpleGate()
        self.shallow_feat1 = nn.Sequential(conv(in_c, width, kernel_size, bias=bias),
                                           DAU(width, kernel_size, reduction, bias=bias, act=act))
        self.info = nn.Conv2d(width, n_feat, 1)
        self.stage1 = Bottneck(in_c, width)
        # self.stage2 = CASNet(n_feat, width, kernel_size, reduction, act, bias,
        #                      num_cab)
        self.concat12 = conv(width * 2, width, kernel_size, bias=bias)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)

        self.conv2 = conv(32, 3, kernel_size=1)

    def forward(self, x3_img):
        x1 = self.shallow_feat1(x3_img)

        x_s1, x1_en, x1_dn = self.stage1(x3_img)

        x2_cat = self.concat12(torch.cat([x1, x1_dn[3]], 1))
        x2_cat = self.conv2(x2_cat)
        x2_cat = x2_cat + x3_img

        return x2_cat

class UNet_my_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_my_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.a = nn.parameter.Parameter(torch.ones(size=1))

        self.Conv1_1 = Conv1x1(128, 64)
        self.Conv1_2 = Conv1x1(256, 128)
        self.Conv1_3 = Conv1x1(512, 256)
        self.Conv1_4 = Conv1x1(1024, 512)
        self.Conv1_5 = Conv1x1(2048, 1024)

        self.SCM1 = SCM(128)
        self.SCM2 = SCM(256)
        self.SCM3 = SCM(512)

        self.CA_1 = ChannelAttention(128)
        self.CA_2 = ChannelAttention(256)
        self.CA_3 = ChannelAttention(512)

    def forward(self, x):
        x_down1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        x_down3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=False)
        x_down4 = F.interpolate(x, scale_factor=0.0625, mode='bilinear', align_corners=False)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x_down1 = self.SCM1(x_down1)
        x2 = torch.cat([x2, x_down1], dim=1)
        x2 = self.Conv1_2(x2)
        x2_CA = self.CA_1(x2)
        x2_ = x2 * x2_CA
        x2 = x2 + x2_

        x3 = self.down2(x2)
        x_down2 = self.SCM2(x_down2)
        x3 = torch.cat([x3, x_down2], dim=1)
        x3 = self.Conv1_3(x3)
        x3_CA = self.CA_2(x3)
        x3_ = x3 * x3_CA
        x3 = x3 + x3_

        x4 = self.down3(x3)
        x_down3 = self.SCM3(x_down3)
        x4 = torch.cat([x4, x_down3], dim=1)
        x4 = self.Conv1_4(x4)
        x4_CA = self.CA_3(x4)
        x4_ = x4 * x4_CA
        x4 = x4 + x4_

        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits