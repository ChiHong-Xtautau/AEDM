import torch
import torch.nn as nn
import torch.nn.functional as F


class NetGen(nn.Module):
    def __init__(self, nz=128, nc=1, img_size=28, ngf=64):
        super(NetGen, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.BatchNorm2d(nc, affine=False),
        )

    def forward(self, z, no_act=False):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)

        img = (0.1*img) + torch.randn_like(img)

        return img


def weights_init(self, m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        if self.args.cuda:
            m.weight.data = torch.where(m.weight.data > 0, m.weight.data, torch.zeros(m.weight.data.shape).cuda())
        else:
            m.weight.data = torch.where(m.weight.data > 0, m.weight.data, torch.zeros(m.weight.data.shape))

