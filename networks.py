import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from hat_arch import HAT,HAT2
import matplotlib.pyplot as plt

device = 'cpu'
up_scale_factor = 4
scale_factor = 1 / up_scale_factor

from utils import show_tensor_images
class HATNet(nn.Module):
    def __init__(self,ch):
        super(HATNet, self).__init__()
        self.net = HAT(upscale=up_scale_factor)
    def forward(self,x):
        return self.net(x)


class UHATNet(nn.Module):
    def __init__(self,ch):
        super(UHATNet, self).__init__()
        self.net = HAT2(upscale=up_scale_factor)
        self.u_estimator = nn.Sequential(
            nn.Upsample(scale_factor=up_scale_factor,mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.ELU(alpha=0.1),
        )

    def forward(self,x):
        sr,fea =  self.net(x)
        u = self.u_estimator(fea) + 0.1
        return sr , u

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.normal_(m.bias, 0.02)
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.normal_(m.bias, 0.02)




class Weight(nn.Module):
    def __init__(self, k):
        super(Weight, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.k = k

    def forward(self, x):
        b, ch, w, h = x.size()

        x = self.conv(x)
        s = w * h  # torch.sum(torch.sum(loss,dim=3),dim=2) #w * h #* ch * b
        sum_x = torch.sum(torch.sum(x, dim=3), dim=2)  # *0.99+0.005
        # sum_x = torch.sum(x)*0.99+0.005

        alpha = F.relu((self.k * s - sum_x) / (s - sum_x))
        alpha = torch.unsqueeze(torch.unsqueeze(alpha, dim=2), dim=3)

        beta = F.relu((sum_x - self.k * s) / sum_x)
        beta = torch.unsqueeze(torch.unsqueeze(beta, dim=2), dim=3)

        w = x + alpha * (1 - x) - beta * x

        # w = 0.99*w + 0.005
        # n = torch.rand_like(w)*0.99 + 0.005

        # v = 10*torch.log(w*n/((1-w)*(1-n)))
        # v_ = torch.abs(v)
        # b = torch.exp((v-v_)/2)/(torch.exp(-(v+v_)/2)+torch.exp((v-v_)/2))
        # return b
        return w

    def setk(self, k):
        self.k = k


class WeightStochastic(nn.Module):
    def __init__(self, k):
        super(WeightStochastic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.k = k

    def forward(self, x, sample_num=1):
        b, ch, w, h = x.size()

        x = self.conv(x)

        #xx = x.clone()
        #his, bin = torch.histogram(xx.cpu(), bins=100, range=(0., 1.))
        #plt.plot(bin[:-1], torch.log10(his))
        #plt.xlabel('conv out')
        #plt.savefig('f/fig_conv_out.jpg')
        #show_tensor_images(x * 2 - 1.0, filename='f/conv_out.jpg')

        s = w * h
        sum_x = torch.sum(torch.sum(x, dim=3), dim=2)  # *0.99+0.005
        # sum_x = torch.sum(x)*0.99+0.005

        alpha = F.relu((self.k * s - sum_x) / (s - sum_x))
        alpha = torch.unsqueeze(torch.unsqueeze(alpha, dim=2), dim=3)

        beta = F.relu((sum_x - self.k * s) / sum_x)
        beta = torch.unsqueeze(torch.unsqueeze(beta, dim=2), dim=3)

        w = x + alpha * (1 - x) - beta * x

        ww = w.clone()
        his, bin = torch.histogram(ww.cpu(), bins=100, range=(0., 1.))
        plt.plot(bin[:-1], torch.log10(his))
        plt.xlabel('fixedsum out')
        plt.savefig('f/fig_fixedsum_out.jpg')
        show_tensor_images(w * 2 - 1.0, filename='f/fixedsum_out.jpg')
        w = 0.999 * w + 0.0005

        if (sample_num == 1):
            n = torch.rand_like(w) * 0.999 + 0.0005

            v = 10 * torch.log(w * n / ((1 - w) * (1 - n)))
            v_ = torch.abs(v)
            b = torch.exp((v - v_) / 2) / (torch.exp(-(v + v_) / 2) + torch.exp((v - v_) / 2))

            return b
        else:
            bs = []
            for i in range(sample_num):
                n = torch.rand_like(w) * 0.999 + 0.0005

                v = 10 * torch.log(w * n / ((1 - w) * (1 - n)))
                v_ = torch.abs(v)
                b = torch.exp((v - v_) / 2) / (torch.exp(-(v + v_) / 2) + torch.exp((v - v_) / 2))
                bs += [b]
            return bs

    def setk(self, k):
        self.k = k


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class RCAN(nn.Module):
    def __init__(self, channel):
        super(RCAN, self).__init__()
        scale = up_scale_factor  # args.scale
        num_features = 64  # args.num_features
        num_rg = 10  # args.num_rg
        num_rcab = 20  # args.num_rcab
        reduction = 16  # args.reduction

        self.sf = nn.Conv2d(channel, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.conv2 = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.upscale(x)
        x = self.conv2(x)
        return x

class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out

class VDSR(nn.Module):
    def __init__(self , n_color) -> None:
        super(VDSR, self).__init__()
        # Input layer
        self.bicubic = nn.Upsample(scale_factor=up_scale_factor, align_corners=False, mode='bicubic')

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_color, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = []
        for _ in range(10):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1), bias=False)

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bicubic(x)
        identity = x
        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, np.sqrt(
                    2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.bicubic = nn.Upsample(scale_factor=up_scale_factor, align_corners=False, mode='bicubic')
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bicubic(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x




def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, n_colors, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = up_scale_factor
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(1.0)
        self.add_mean = MeanShift(1.0, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        #x = self.add_mean(x)
        #print(torch.mean(x),torch.min(x),torch.max(x))

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

