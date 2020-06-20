import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from skimage import img_as_ubyte


class Discriminator(nn.Module):
    def __init__(self, input_shape=(6, 128, 128)):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class channel_ap(nn.Module):
    def __init__(self):
        super(channel_ap, self).__init__()

    def forward(self, x):
        y = torch.mean(x, dim=1, keepdim=True)
        return y


class channel_mp(nn.Module):
    def __init__(self):
        super(channel_mp, self).__init__()

    def forward(self, x):
        y = torch.max(x, dim=1, keepdim=True)
        return y[0]


class ResidualBlock(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        )
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        y = self.main(x)
        z = self.act(y + x)
        return z


class SAUnit(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inter_channel=2):
        super(SAUnit, self).__init__()
        self.in_ch = in_ch
        self.Ch_AP = channel_ap()
        self.Ch_MP = channel_mp()

        self.C = nn.Sequential(
            nn.Conv2d(inter_channel, 1, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid()
        )
        self.W = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        ap = self.Ch_AP(x)
        mp = self.Ch_MP(x)
        pp = torch.cat((ap, mp), dim=1)
        heatmap = self.C(pp).repeat(1, self.in_ch, 1, 1)

        y = heatmap * self.W(x) + x
        return y


class CAUnit(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inter_channel=4):
        super(CAUnit, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.C1 = nn.Sequential(
            nn.Conv2d(in_ch, inter_channel, kernel_size=3, padding=1, stride=1),
            nn.PReLU(inter_channel)
        )

        self.C2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_ch, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.C1(y)
        y2 = self.C2(y1)

        return x * y2


class NTBlock(nn.Module):
    def __init__(self, inter_ch=64, SA_ch=2, CA_ch=4):
        super(NTBlock, self).__init__()

        self.module = nn.Sequential(
            ResidualBlock(inter_ch, inter_ch, 3, 1, 1),
            ResidualBlock(inter_ch, inter_ch, 3, 1, 1),
            SAUnit(inter_ch, inter_ch, SA_ch),
            ResidualBlock(inter_ch, inter_ch, 3, 1, 1),
            ResidualBlock(inter_ch, inter_ch, 3, 1, 1),
            CAUnit(inter_ch, inter_ch, CA_ch)
        )

    def forward(self, x):
        y = self.module(x)
        return y + x


class NTGAN(nn.Module):
    def __init__(self, im_ch=3, inter_ch=64, SA_ch=2, CA_ch=4):
        super(NTGAN, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = MeanShift(1, rgb_mean, rgb_std)
        self.add_mean = MeanShift(1, rgb_mean, rgb_std, 1)

        self.in_conv = nn.Sequential(
            nn.Conv2d(im_ch, inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(inter_ch)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(inter_ch * 2, inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(inter_ch),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(inter_ch),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(inter_ch),
            nn.Conv2d(inter_ch, im_ch, kernel_size=3, padding=1, stride=1)
        )

        self.NTB1 = NTBlock(inter_ch=inter_ch, SA_ch=SA_ch, CA_ch=CA_ch)
        self.NTB2 = NTBlock(inter_ch=inter_ch, SA_ch=SA_ch, CA_ch=CA_ch)
        self.NTB3 = NTBlock(inter_ch=inter_ch, SA_ch=SA_ch, CA_ch=CA_ch)
        self.NTB4 = NTBlock(inter_ch=inter_ch, SA_ch=SA_ch, CA_ch=CA_ch)

        self.noise_level_branch = nn.Sequential(
            nn.Conv2d(im_ch, inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(inter_ch),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(inter_ch),
            nn.AvgPool2d(2),
            nn.Conv2d(inter_ch, 2 * inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(2 * inter_ch),
            nn.Conv2d(2 * inter_ch, 2 * inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(2 * inter_ch),
            nn.Conv2d(2 * inter_ch, 2 * inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(2 * inter_ch),
            nn.AvgPool2d(2),
            nn.Conv2d(2 * inter_ch, 4 * inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(4 * inter_ch),
            nn.Conv2d(4 * inter_ch, 4 * inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(4 * inter_ch),
            nn.Conv2d(4 * inter_ch, 4 * inter_ch, kernel_size=3, padding=1, stride=1),
            nn.PReLU(4 * inter_ch),
            nn.ConvTranspose2d(4 * inter_ch, inter_ch, kernel_size=4, padding=0, stride=4)
        )

    def forward(self, x, nlm):
        eps = torch.randn_like(nlm)
        nlm_r = nlm * eps

        s = self.sub_mean(x)
        x_im = self.in_conv(s)

        y_im1 = self.NTB1(x_im)
        y_im2 = self.NTB2(y_im1)
        y_im3 = self.NTB3(y_im2)
        y_im4 = self.NTB3(y_im3)

        y_nlm = self.noise_level_branch(nlm_r)

        z = self.out_conv(torch.cat((x_im + y_im4, y_nlm), dim=1))
        z = self.add_mean(z)
        z = z + x + nlm_r
        return z


