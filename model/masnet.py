import torch
from torchvision import models as resnet_model
from torch import nn
from torchvision.models import swin_b, Swin_B_Weights, swin_t, Swin_T_Weights
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn.functional as F
from timm.models.helpers import named_apply
from functools import partial
from model.pvt import pvt_v2_b2

import math


class MemoryAugmentation(nn.Module):
    def __init__(self, M=10, H=88, W=88):
        super(MemoryAugmentation, self).__init__()
        self.M = nn.Parameter(torch.FloatTensor(M, H, W))
        nn.init.xavier_normal_(self.M)

    def forward(self, x):
        # print(f'x.shape:{x.shape}, self.M.shape:{self.M.shape}')
        score = torch.softmax(torch.einsum("bchw,mhw->bcm", x, self.M), dim=-1)
        value = torch.einsum("bcm,mhw->bchw", score, self.M)
        return value


class ma(nn.Module):
    def __init__(self, M=10, H=88, W=88):
        super(ma, self).__init__()
        self.meaug = MemoryAugmentation(M=M, H=H, W=W)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        resudiual = x
        x = self.meaug(x)
        return x + resudiual


class cb(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(cb, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class conv_bn_relu(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, skip=False,
                 inplace=True):
        super(conv_bn_relu, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class cs(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(cs, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            cb(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            cb(in_channel, out_channel, 1),
            cb(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            cb(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            cb(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            cb(in_channel, out_channel, 1),
            cb(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            cb(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            cb(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            cb(in_channel, out_channel, 1),
            cb(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            cb(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            cb(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = cb(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = cb(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class clff(nn.Module):

    def __init__(self, channel):
        super(clff, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = cb(channel, channel, 3, padding=1)
        self.conv_upsample2 = cb(channel, channel, 3, padding=1)
        self.conv_upsample3 = cb(channel, channel, 3, padding=1)
        self.conv_upsample4 = cb(channel, channel, 3, padding=1)
        self.conv_upsample5 = cb(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = cb(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = cb(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = cb(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 640, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, channel, r=7):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ms(nn.Module):
    def __init__(self, dim_in, up=None):
        super(ms, self).__init__()
        self.up = up
        self.res = nn.Sequential(
            conv_bn_relu(dim_in=dim_in, dim_out=dim_in * 2, kernel_size=1),
            conv_bn_relu(dim_in=dim_in * 2, dim_out=dim_in * 2, kernel_size=3),
            conv_bn_relu(dim_in=dim_in * 2, dim_out=dim_in, kernel_size=1)
        )
        self.conv = conv_bn_relu(dim_in=9, dim_out=dim_in, kernel_size=1)

    def forward(self, x, origin_x, pred):
        if self.up == True:
            pred = F.interpolate(input=pred, scale_factor=2.0, mode='bilinear', align_corners=False)

        score = torch.sigmoid(pred)
        dist = (score - 0.5) ** 2
        att = 1 - (dist / 0.25)
        att = origin_x * self.conv(att) + x
        shout = att
        out = self.res(att)
        out = out + shout

        return out


class maf(nn.Module):
    def __init__(self, in_channel, out_channel, exp_ratio=1.0):
        super(maf, self).__init__()

        mid_channel = in_channel * exp_ratio

        self.DWConv = conv_bn_relu(mid_channel, mid_channel, kernel_size=3, groups=out_channel // 2)
        self.DWConv3x3 = conv_bn_relu(in_channel // 4, in_channel // 4, kernel_size=3, groups=in_channel // 4)
        self.DWConv5x5 = conv_bn_relu(in_channel // 4, in_channel // 4, kernel_size=5, groups=in_channel // 4)
        self.DWConv7x7 = conv_bn_relu(in_channel // 4, in_channel // 4, kernel_size=7, groups=in_channel // 4)
        self.PWConv1 = conv_bn_relu(in_channel, mid_channel, kernel_size=1)
        self.PWConv2 = conv_bn_relu(mid_channel, out_channel, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channel)
        self.Maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        channels = x.size(1)
        channels_per_part = channels // 4
        x1 = x[:, :channels_per_part, :, :]
        x2 = x[:, channels_per_part:2 * channels_per_part, :, :]
        x3 = x[:, 2 * channels_per_part:3 * channels_per_part, :, :]
        x4 = x[:, 3 * channels_per_part:, :, :]

        x1 = self.Maxpool(x1)
        x2 = self.DWConv3x3(x2)
        x3 = self.DWConv5x5(x3)
        x4 = self.DWConv7x7(x4)

        x2 = x1 * x2
        x3 = x2 * x3
        x4 = x3 * x4
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.PWConv1(x)
        x = x + self.DWConv(x)
        x = self.PWConv2(x)
        x = x + shortcut

        return x


class gf(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(gf, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Masnet(nn.Module):
    def __init__(self, num_classes=9, channels=[640, 256, 128], ):
        super(Masnet, self).__init__()
        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=True)

        self.backbone = pvt_v2_b2()

        path = '../pvt_v2_b2.pth'
        save_model = torch.load(path, weights_only=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.conv1 = cb(128, 64, 1)
        self.conv2 = cb(256, 128, 1)
        self.conv3 = cb(512, 320, 1)

        self.SE1 = SEBlock(128)
        self.SE2 = SEBlock(256)
        self.SE3 = SEBlock(640)

        self.s3 = ma(H=14, W=14)
        self.s2 = ma(H=28, W=28)
        self.s1 = ma(H=56, W=56)

        self.cs1 = cs(128, 32)
        self.cs2 = cs(256, 32)
        self.cs3 = cs(640, 32)

        self.agg = clff(32)

        self.CA3 = CAB(640)
        self.CA2 = CAB(256)
        self.CA1 = CAB(128)

        self.SA = SAB()

        self.maf1 = maf(in_channel=channels[0], out_channel=channels[0], exp_ratio=2)
        self.maf2 = maf(in_channel=channels[1], out_channel=channels[1], exp_ratio=2)
        self.maf3 = maf(in_channel=channels[2], out_channel=channels[2], exp_ratio=2)

        self.msm1 = ms(dim_in=channels[0], up=False)
        self.msm2 = ms(dim_in=channels[1], up=True)
        self.msm3 = ms(dim_in=channels[2], up=True)

        self.down = nn.MaxPool2d(4)
        self.up2 = nn.ConvTranspose2d(640, 256, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.ag3 = gf(F_g=640, F_l=640, F_int=640 // 2, kernel_size=3, groups=640 // 2)
        self.ag2 = gf(F_g=256, F_l=256, F_int=256 // 2, kernel_size=3, groups=256 // 2)
        self.ag1 = gf(F_g=128, F_l=128, F_int=128 // 2, kernel_size=3, groups=128 // 2)

        self.out_head4 = nn.Conv2d(640, num_classes, 1)
        self.out_head3 = nn.Conv2d(640, num_classes, 1)
        self.out_head2 = nn.Conv2d(256, num_classes, 1)
        self.out_head1 = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        # pvt

        p1, p2, p3, p4 = self.backbone(x)

        # resnet

        r = self.resnet.conv1(x)
        r = self.resnet.bn1(r)
        r = self.resnet.relu(r)

        r = self.resnet.layer1(r)

        r1 = self.resnet.layer2(r)
        r2 = self.resnet.layer3(r1)
        r3 = self.resnet.layer4(r2)

        r1 = self.conv1(r1)
        r2 = self.conv2(r2)
        r3 = self.conv3(r3)

        all1 = torch.cat([p1, r1], dim=1)
        all2 = torch.cat([p2, r2], dim=1)
        all3 = torch.cat([p3, r3], dim=1)

        all1 = self.SE1(all1)
        all2 = self.SE2(all2)
        all3 = self.SE3(all3)

        all1 = self.s1(all1)
        all2 = self.s2(all2)
        all3 = self.s3(all3)


        c1 = self.cs1(all1)
        c2 = self.cs2(all2)
        c3 = self.cs3(all3)

        out4 = self.agg(c3, c2, c1)
        o4 = self.down(out4)

        pre4 = self.out_head4(o4)

        out4 = self.out_head4(out4)
        out4 = F.interpolate(out4, scale_factor=4, mode='bilinear')

        dd3 = self.ag3(o4, all3)
        dd3 = dd3 + o4

        d3 = self.CA3(dd3) * dd3
        d3 = self.SA(d3) * d3
        d3 = self.maf1(d3)
        d3 = self.msm1(d3, all3, pre4)

        out3 = self.out_head3(d3)
        pre3 = out3
        out3 = F.interpolate(out3, scale_factor=16, mode='bilinear')

        d2 = self.up2(d3)
        x2 = self.ag2(d2, all2)
        d2 = d2 + x2

        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.maf2(d2)
        d2 = self.msm2(d2, all2, pre3)

        out2 = self.out_head2(d2)
        pre2 = out2
        out2 = F.interpolate(out2, scale_factor=8, mode='bilinear')

        d1 = self.up1(d2)
        x1 = self.ag1(d1, all1)
        d1 = d1 + x1

        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.maf3(d1)
        d1 = self.msm3(d1, all1, pre2)

        out1 = self.out_head1(d1)
        out1 = F.interpolate(out1, scale_factor=4, mode='bilinear')

        return out4, out3, out2, out1



if __name__ == '__main__':
    x = torch.rand(5, 3, 224, 224)
    model = Masnet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"params: {total_params:,} ({total_params/1e6:.2f}M)")  
    y = model(x)







