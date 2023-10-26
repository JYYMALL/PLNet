import torch
import torch.nn as nn
from hough.dht import DHT_Layer
import torch.nn.functional as F

from net.Res2Net import res2net50_v1b_26w_4s


class ConvBPR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBPR, self).__init__()
        # (k_size+(k_size-1)*(dilation-1))//2
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.p_relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.p_relu(x)

        return x


class FusionAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FusionAttention, self).__init__()
        self.cat = Conv1x1(in_channel, out_channel)
        self.smooth = Conv1x1(out_channel, out_channel)
        self.att_conv = nn.Sequential(Conv1x1(out_channel, 2), nn.Softmax(dim=1))

    def forward(self, x1, x2):
        if x1.size()[2:] != x2.size()[2:]:
            x2 = F.interpolate(x2, x1.size()[2:], mode='bilinear', align_corners=False)
        x12 = self.cat(torch.cat([x1, x2], dim=1))
        att = self.att_conv(x12)
        att = torch.chunk(att, 2, dim=1)
        out = x1 * att[0] + x2 * att[1]
        return self.smooth(out)


class MCFF(nn.Module):
    def __init__(self, in_channel):
        super(MCFF, self).__init__()
        channel = in_channel // 4
        self.chunk1 = Conv1x1(in_channel, channel)
        self.chunk2 = Conv1x1(in_channel, channel)
        self.chunk3 = Conv1x1(in_channel, channel)
        self.chunk4 = Conv1x1(in_channel, channel)
        self.dconv3_1 = ConvBPR(channel, channel, 3)
        self.dconv3_2 = ConvBPR(channel, channel, 3, dilation=2)
        self.dconv3_3 = ConvBPR(channel, channel, 3, dilation=3)
        self.dconv3_4 = ConvBPR(channel, channel, 3, dilation=4)
        self.conv3_3 = ConvBPR(in_channel, in_channel, 3)
        # self.smooth = Conv1x1(in_channel, in_channel)

    def forward(self, x):
        x1, x2, x3, x4 = self.chunk1(x), self.chunk2(x), self.chunk3(x), self.chunk4(x)
        x1 = self.dconv3_1(x1)
        x2 = self.dconv3_2(x2 + x1)
        x3 = self.dconv3_3(x3 + x2)
        x4 = self.dconv3_4(x4 + x3)
        out = self.conv3_3(torch.cat([x1, x2, x3, x4], dim=1))
        return x + out


class LFE(nn.Module):
    def __init__(self, ratio, channel=256):
        super(LFE, self).__init__()
        self.conv1 = ConvBPR(channel, channel, 3)
        self.conv2 = ConvBPR(channel, channel, 3)
        self.att_block = FusionAttention(channel * 2, channel)
        self.ht_block = nn.Sequential(
            DHT_Layer(input_dim=1, dim=1, numAngle=128, numRho=128 // ratio)
        )

        self.bnr = nn.Sequential(nn.BatchNorm2d(channel), nn.PReLU())
        self.final = nn.Conv2d(channel, 1, 1)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv3 = ConvBPR(1, 32, 3)

    def forward(self, x):
        fore = self.conv1(x)
        back = 1 - self.conv2(x)
        back = self.bnr(back)
        out = self.att_block(fore, back)
        o1 = self.final(out)
        ht = self.ht_block(o1)

        return out, o1, ht


class LFA(nn.Module):
    def __init__(self, channel):
        super(LFA, self).__init__()
        self.conv1 = nn.Sequential(ConvBPR(channel // 4, channel // 4, kernel_size=3),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.PReLU())
        self.conv2 = nn.Sequential(ConvBPR(channel // 4, channel // 4, kernel_size=3),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.PReLU())
        self.conv3 = nn.Sequential(ConvBPR(channel // 4, channel // 4, kernel_size=3),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.PReLU())
        self.conv4 = nn.Sequential(ConvBPR(channel // 4, channel // 4, kernel_size=3),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.PReLU())

        self.cat = ConvBPR(channel, channel, 3)
        self.wei = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Softmax(dim=1))

    def forward(self, x, att):
        if x.size()[2:] != att.size()[2:]:
            att = F.interpolate(att, x.size()[2:], mode='bilinear', align_corners=False)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        att = torch.sigmoid(att)
        a1 = self.conv1(x1) * att
        a2 = self.conv2(x2) * att
        a3 = self.conv3(x3) * att
        a4 = self.conv4(x4) * att
        wei = self.wei(torch.cat((a1, a2, a3, a4), 1))
        w1, w2, w3, w4 = torch.chunk(wei, 4, dim=1)
        xw1 = x1 * w1
        xw2 = x2 * w2
        xw3 = x3 * w3
        xw4 = x4 * w4
        return self.cat(torch.cat((xw1, xw2, xw3, xw4), dim=1))


class PLNet(nn.Module):
    def __init__(self):
        super(PLNet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.reduce1 = ConvBPR(256, 256)
        self.reduce2 = ConvBPR(512, 256)
        self.reduce3 = ConvBPR(1024, 256)
        self.reduce4 = ConvBPR(2048, 256)

        self.lfe1 = LFE(1)
        self.lfe2 = LFE(2)
        self.lfe3 = LFE(4)
        self.lfe4 = LFE(8)

        self.lfa1 = LFA(256)
        self.lfa2 = LFA(256)
        self.lfa3 = LFA(256)

        self.mcff1 = MCFF(256)
        self.mcff2 = MCFF(256)
        self.mcff3 = MCFF(256)
        self.mcff4 = MCFF(256)

        self.fusion3 = Conv1x1(512, 256)
        self.fusion2 = Conv1x1(512, 256)
        self.fusion1 = Conv1x1(512, 256)

        self.ht_fusion = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.resnet(x)

        x1 = self.reduce1(x1)  # 256
        x2 = self.reduce2(x2)  # 256
        x3 = self.reduce3(x3)  # 256
        x4 = self.reduce4(x4)  # 256
        x4 = self.mcff4(x4)
        x3 = self.mcff3(x3)
        x2 = self.mcff2(x2)
        x1 = self.mcff1(x1)

        l4, o4, ht4 = self.lfe4(x4)
        l4 = F.interpolate(l4, size=x3.size()[2:], mode='bilinear', align_corners=False)
        f3 = self.fusion3(torch.cat([x3, l4], dim=1))
        g3 = self.lfa3(f3, o4)

        l3, o3, ht3 = self.lfe3(g3)
        l3 = F.interpolate(l3, size=x2.size()[2:], mode='bilinear', align_corners=False)
        f2 = self.fusion2(torch.cat([x2, l3], dim=1))
        g2 = self.lfa2(f2, o3)

        l2, o2, ht2 = self.lfe2(g2)
        l2 = F.interpolate(l2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        f1 = self.fusion1(torch.cat([x1, l2], dim=1))
        g1 = self.lfa1(f1, o2)

        l1, o1, ht1 = self.lfe1(g1)

        ht4 = F.interpolate(ht4, size=ht1.size()[2:], mode='bilinear', align_corners=False)
        ht3 = F.interpolate(ht3, size=ht1.size()[2:], mode='bilinear', align_corners=False)
        ht2 = F.interpolate(ht2, size=ht1.size()[2:], mode='bilinear', align_corners=False)
        ht_out = self.ht_fusion(torch.cat([ht4, ht3, ht2, ht1], dim=1))

        o4 = F.interpolate(o4, size=x.size()[2:], mode='bilinear', align_corners=False)
        o3 = F.interpolate(o3, size=x.size()[2:], mode='bilinear', align_corners=False)
        o2 = F.interpolate(o2, size=x.size()[2:], mode='bilinear', align_corners=False)
        logit = F.interpolate(o1, size=x.size()[2:], mode='bilinear', align_corners=False)

        return o4, o3, o2, logit, ht_out
