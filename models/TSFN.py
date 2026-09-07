import math
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        # HSI branch
        self.scale_ratio = config[config['train_dataset']]['factor']
        self.n_bands = config[config['train_dataset']]['spectral_bands']
        self.n_select_bands = 3
        
        self.input_1 = nn.Sequential(
            nn.Conv2d(self.n_bands, 64, kernel_size=3, padding=1),
            nn.PReLU())
        self.residual_layers_1 = nn.Sequential(*[nn.Sequential(ResidualBlock(64)) for _ in range(4)])
        self.output_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))

        # RGB branch
        self.input_2 = nn.Sequential(
            nn.Conv2d(self.n_select_bands, 64, kernel_size=3, padding=1),
            nn.PReLU())
        self.residual_layers_2 = nn.Sequential(*[nn.Sequential(ResidualBlock(64)) for _ in range(4)])
        self.output_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))

        # Fusion
        self.layer_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, self.n_bands, kernel_size=3, padding=1))

    def forward(self, input_hsi, input_rgb):
        
        input_hsi = F.interpolate(input_hsi, scale_factor=self.scale_ratio, mode='bilinear')
        out1_1 = self.input_1(input_hsi)
        out2_1 = self.residual_layers_1(out1_1)
        out3_1 = self.output_1(out2_1)
        
        
        out1_2 = self.input_2(input_rgb)
        out2_2 = self.residual_layers_2(out1_2)
        out3_2 = self.output_2(out2_2)
        #print(out3_1.shape)
        #print(out3_2.shape)
        out4 = self.layer_fusion(torch.cat((out3_1, out3_2),1))
        out = torch.add(input_hsi, out4)
        return {"pred": out}

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


# # test
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# hsi = torch.ones(1,31,256,256).to(device)
# rgb = torch.ones(1,3,256,256).to(device)

# model = Net(HSI_num_residuals=12, RGB_num_residuals=12).to(device)
# output = model(hsi, rgb)
# print(output.size())