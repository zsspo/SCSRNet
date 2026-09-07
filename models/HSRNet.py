import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class HSRNet(nn.Module):

    def __init__(self,
                 config):
        super(HSRNet, self).__init__()
        self.scale_ratio = config[config['train_dataset']]['factor']
        self.n_bands = config[config['train_dataset']]['spectral_bands']
        self.n_select_bands = config[config['train_dataset']]['msi_bands']

        # Channel Attention
        self.CA_GAP = nn.AdaptiveAvgPool2d(1)  #全局平均池化
        self.CA_1x1_1conv = nn.Sequential(
            nn.Conv2d(in_channels=self.n_bands, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.CA_1x1_Sconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.n_bands, kernel_size=1, stride=1, padding=0),  # B,S,1,1
            nn.Sigmoid(),
        )


        # Spatial Attention
        # 注意卷积之前先F.pad一下  在右侧pad 1列
        self.SA_6x6_1conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=6, stride=1, padding=2),   # B,1,H,W
            nn.Sigmoid(),
        )

        # 降采样 MSI
        self.downsample = nn.Conv2d(in_channels=self.n_select_bands, out_channels=self.n_select_bands, kernel_size=5, stride=3, padding=1)
        #self.downsample = nn.Conv2d(in_channels=self.n_select_bands, out_channels=self.n_select_bands, kernel_size=6, stride=4, padding=1)
        # PixelShuffle
        self.pixelshuffle_onestep = nn.Sequential(
            nn.Conv2d(in_channels=3 + self.n_bands, out_channels=self.n_bands * (self.scale_ratio**2), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.PixelShuffle(self.scale_ratio),
        )

        # ResNet 部分
        self.conv_3x3_64 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_bands + 3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )
        # 用完 记得 add
        self.ResNetBlock = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.conv_3x3_S = nn.Conv2d(in_channels=64, out_channels=self.n_bands, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # print('conv2d')
            nn.init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_lr, x_hr):
        # 计算 Channel Attention
        CA = self.CA_GAP(x_lr)
        CA = self.CA_1x1_1conv(CA)
        CA = self.CA_1x1_Sconv(CA)

        # 计算 Spatial Attention
        SA = torch.mean(x_hr, dim=1)
        SA = F.pad(SA, (0, 1, 0, 1, 0, 0,), mode='constant', value=0)
        SA = torch.unsqueeze(SA, 1)
        SA = self.SA_6x6_1conv(SA)


        # downsample
        downsample = self.downsample(x_hr)
        #R, G, B = torch.split(downsample, [1, 1, 1], dim=1)
        R, G, B, _ = torch.split(downsample, [1, 1, 1, 1], dim=1)
        lr1, lr2 = torch.split(x_lr, [(int(self.n_bands/2)), (math.ceil(self.n_bands/2))], dim=1)
        C0 = torch.concat((R, lr1, G, lr2, B), dim=1)

        # PixelShuffle
        YPS = self.pixelshuffle_onestep(C0)

        #R, G, B = torch.split(x_hr, [1, 1, 1], dim=1)
        R, G, B, _ = torch.split(x_hr, [1, 1, 1, 1], dim=1)
        lr1, lr2 = torch.split(YPS, [(int((self.n_bands) / 2)), (math.ceil((self.n_bands) / 2))], dim=1)
        C1 = torch.concat((R, lr1, G, lr2, B), dim=1)

        C1 = self.conv_3x3_64(C1)
        C1 = self.ResNetBlock(C1) + C1

        C1_S = C1 * SA
        C1_S = self.conv_3x3_S(C1_S)

        OUT = C1_S * CA

        # add
        lr_up = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode = 'bicubic')
        OUT = OUT + lr_up
        return {"pred": OUT}



# model = HSRNet(4,3,128)
