import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

# We use GELU as the activation function, which is commonly used in modern attention-based networks.
ACTIVATION = nn.GELU

# --- 1. Large Kernel Attention (LKA) Block (Single Scale) ---
class LKA_Block(nn.Module):
    """
    Implements the core LKA decomposition as described in the paper, which
    consists of a Depthwise Conv (Local), a Depthwise Dilated Conv (Long-Range),
    and a Pointwise Conv (Channel Aggregation) to compute the attention map,
    followed by element-wise multiplication with a content feature.

    LKA Parameters:
    - k_i: Total conceptual large kernel size (e.g., 7, 21, 35)
    - d_i: Dilation rate (e.g., 2, 3, 4)
    """
    def __init__(self, channels, k_i, d_i):
        super().__init__()
        
        # 1. Attention Map Path (X0 in paper)
        # 1.1. Local Aggregation: Depthwise Conv (kernel size 2*d_i - 1)
        local_kernel_size = 2 * d_i - 1
        self.conv_dw_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=local_kernel_size, padding=local_kernel_size//2, groups=channels),
            ACTIVATION()
        )

        # 1.2. Long-Range Dependency: Depthwise Dilated Conv (DWDConv, kernel size ceil(k_i/d_i), dilation d_i)
        long_range_kernel_size = math.ceil(k_i / d_i)
        if long_range_kernel_size%2 ==0:
            long_range_kernel_size -= 1
        # Note: Padding must be calculated to keep feature map size constant
        padding = (long_range_kernel_size - 1) * d_i // 2
        self.conv_dwd_long = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=long_range_kernel_size, padding=padding, dilation=d_i, groups=channels),
            ACTIVATION()
        )

        # 1.3. Channel Aggregation: Pointwise Conv (PWConv)
        self.conv_pw_attn = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Attention Map Path: DWConv_local -> DWDConv_long -> PWConv_attn (produces attention map A)
        
        # 2. Content Feature Path (Conv_dw(X) in paper, used for local information)
        # The paper suggests a DWConv with kernel size ceil(k_i/d_i) for content feature.
        self.conv_dw_content = nn.Conv2d(channels, channels, kernel_size=long_range_kernel_size, padding=long_range_kernel_size//2, groups=channels)

       


    def forward(self, x):
        # 1. Generate Attention Map (A)
        x_attn = self.conv_dw_local(x)
        x_attn = self.conv_dwd_long(x_attn)
        A = self.conv_pw_attn(x_attn)

        # 2. Generate Content Feature (V)
        V = self.conv_dw_content(x)

        # 3. Element-wise product (Attention * Content)
        # X1 = X0 (Attention Map) * Conv_dw(X) (Content Feature)
        x_out = A * V

        return x_out

# --- 2. Multiscale Large Kernel Attention Module (MLKAM) ---
class MLKAM(nn.Module):
    """
    Multiscale Large Kernel Attention Module.
    Splits features into F0 and G, further splits G into G1, G2, G3 for
    three different LKA scales, and then fuses the results.
    """
    def __init__(self, channels):
        super().__init__()
        
        # Layer Normalization, inserted into MLKAM
        self.layer_norm = nn.LayerNorm(channels)

        # PWConv to double channels (C -> 2C)
        self.pw_conv_double = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        
        # PWConv for F0 path (C -> C)
        self.pw_conv_f0 = nn.Conv2d(channels, channels, kernel_size=1)

        # PWConv to maintain output dimensions (C -> C)
        self.pw_conv_out = nn.Conv2d(channels, channels, kernel_size=1)
        
        
        # LKA configurations (k_i, d_i) -> {7, 2), {21, 3}, {35, 4}
        c_div3 = channels // 3
        
        self.lka1 = LKA_Block(c_div3, k_i=7, d_i=2)
        self.lka2 = LKA_Block(c_div3, k_i=21, d_i=3)
        self.lka3 = LKA_Block(c_div3, k_i=35, d_i=4)

    def forward(self, x):
        # LN(X)
        x_ln = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Conv_pw(LN(X)) -> 2C channels
        x_pc = self.pw_conv_double(x_ln)
        
        # Split into [F0, G] (each C channels)
        F0, G = torch.chunk(x_pc, 2, dim=1)
        
        # Split G into [G1, G2, G3] (each C/3 channels)
        G1, G2, G3 = torch.chunk(G, 3, dim=1)
        
        # Apply multiscale LKA
        A1 = self.lka1(G1)
        A2 = self.lka2(G2)
        A3 = self.lka3(G3)
        
        # Concatenate results G_m
        Gm = torch.cat([A1, A2, A3], dim=1) # C channels
        
        # Final fusion (simplified based on diagram/common practices)
        # F = A * Conv_pw(Conv_pw(F0) * Gm) is the text equation.
        # F = PWConv(PWConv(F0) * Gm) is the diagram interpretation.
        
        F0_processed = self.pw_conv_f0(F0)
        
        # Element-wise product
        F_tmp = F0_processed * Gm # C channels
        
        # Final channel aggregation and scaling
        F_out = self.pw_conv_out(F_tmp) 
        
        return F_out

# --- 3. Spatial Information Aggregation Module (SIAM) ---
class SIAM(nn.Module):
    """
    Spatial Information Aggregation Module (Fig. 1(d)).
    Uses a large kernel Depthwise Conv (k=7) for efficient spatial feature extraction.
    """
    def __init__(self, channels):
        super().__init__()
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(channels)
        
        # PWConv to prepare for splitting (C -> C)
        self.pw_conv_in = nn.Conv2d(channels, channels, kernel_size=1)
        
        # DWConv (k=7) for spatial features on F1 path (C/2 -> C/2)
        c_half = channels // 2
        self.dw_conv_k7 = nn.Conv2d(c_half, c_half, kernel_size=7, padding=3, groups=c_half)
        
        # PWConv to aggregate and restore channel size (C/2 -> C)
        self.pw_conv_out = nn.Conv2d(c_half, channels, kernel_size=1)
        
    def forward(self, x):
        # LN(F)
        x_ln = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Conv_pw(LN(F))
        x_pc = self.pw_conv_in(x_ln)
        
        # Split into F1, F2 (each C/2 channels)
        F1, F2 = torch.chunk(x_pc, 2, dim=1)
        
        # Spatial feature extraction on F1, followed by element-wise product with F2
        F_dw = self.dw_conv_k7(F1)
        F_s_tmp = F_dw * F2 # (C/2 channels)
        
        # Final channel aggregation
        F_s = self.pw_conv_out(F_s_tmp)
        return F_s

# --- 4. Spectral Attention Module (SPAM) ---
class SPAM(nn.Module):
    """
    Spectral Attention Module (Fig. 1(e)).
    Channel attention mechanism using 1x1 convolutions (instead of MLP).
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(channels)

        c_reduced = channels // reduction_ratio
        
        # 1x1 Conv (reduce channels)
        self.conv_reduce = nn.Conv2d(channels, c_reduced, kernel_size=1)
        self.relu = ACTIVATION()
        
        # 1x1 Conv (restore channels)
        self.conv_restore = nn.Conv2d(c_reduced, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Global Average Pooling (AvgPool)
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Squeeze (Conv_1x1 -> ReLU)
        x_s = self.relu(self.conv_reduce(x_avg))
        
        # Excitation (Conv_1x1 -> Sigmoid) -> Attention vector S
        S = self.sigmoid(self.conv_restore(x_s))
        
        # Element-wise product (F_out = F_in * S)
        # Note: Since SPAM is the last step in HAB, it should produce the feature map
        # that will be added back. We assume F_out = F_in * S based on standard CA.
        return x * S
        
# --- 5. Hybrid Attention Block (HAB) ---
class HAB(nn.Module):
    """
    Hybrid Attention Block (Fig. 1(b)).
    Contains MLKAM, SIAM, and SPAM with cascaded residual connections.
    F_out = F_in + MLKAM(F_in) + SIAM(F_1) + SPAM(F_2)
    """
    def __init__(self, channels):
        super().__init__()
        
        self.mlkam = MLKAM(channels)
        self.siam = SIAM(channels)
        self.spam = SPAM(channels)
        
    def forward(self, x):
        # F_1 = F_in + MLKAM(F_in)
        F_mlkam = self.mlkam(x)
        F_1 = x + F_mlkam
        
        # F_2 = F_1 + SIAM(F_1)
        F_siam = self.siam(F_1)
        F_2 = F_1 + F_siam
        
        # F_out = F_2 + SPAM(F_2)
        F_spam = self.spam(F_2) # F_2 * S
        F_out = F_2 + F_spam
        
        return F_out

# --- 6. MLKAF-Net (Overall Architecture) ---
class MLKAF_Net(nn.Module):
    """
    MLKAF-Net: Multiscale Large Kernel Attention Network for HSI-MSI Fusion (Fig. 1(a)).
    """
    def __init__(self, config, C=66, num_habs=7):
        """
        :param L_hsi: Number of spectral bands in LR-HSI (e.g., 102)
        :param L_msi: Number of spectral bands in HR-MSI (e.g., 5)
        :param C: Channel dimension for deep feature extraction (e.g., 64)
        :param num_habs: Number of stacked HABs (e.g., 9, 4, or 7 depending on dataset)
        """
        super().__init__()

        self.scale_ratio = config[config['train_dataset']]['factor']
        self.n_select_bands = config[config['train_dataset']]['msi_bands']

        self.L_hsi = config[config['train_dataset']]['spectral_bands']
        self.L_msi = self.n_select_bands = config[config['train_dataset']]['msi_bands']
        # 1. Shallow Feature Extraction
        # Input: Concatenated feature (LR-HSI upsampled + HR-MSI) -> L_hsi + L_msi channels
        # Output: C channels
        self.shallow_conv = nn.Conv2d(self.L_hsi + self.L_msi, C, kernel_size=3, padding=1)
        
        # 2. Deep Feature Extraction (Stacks of HABs)
        self.deep_feature_extractor = nn.Sequential(
            *[HAB(C) for _ in range(num_habs)]
        )
        
        # 3. Feature Integration
        # Output: Residual map R (C channels -> L_hsi channels)
        self.feature_integration = nn.Conv2d(C, self.L_hsi, kernel_size=1)

    def forward(self, lr_hsi, hr_msi):
        """
        :param lr_hsi: Low-Resolution HSI (B, L_hsi, h, w)
        :param hr_msi: High-Resolution MSI (B, L_msi, H, W)
        :return: High-Resolution HSI (B, L_hsi, H, W)
        """
        
        H, W = hr_msi.shape[-2:]
        
        # 1. Upsample LR-HSI to HR-MSI size (Z^ in paper, H x W)
        # Using bicubic interpolation mode as mentioned in the paper
        hsi_upsampled = F.interpolate(lr_hsi, scale_factor=self.scale_ratio, mode='bicubic', align_corners=False)
        
        # 2. Concatenate upsampled HSI and HR-MSI (Z^ || Y)
        F_concat = torch.cat([hsi_upsampled, hr_msi], dim=1) # (B, L_hsi + L_msi, H, W)
        
        # 3. Shallow Feature Extraction (E)
        E = self.shallow_conv(F_concat) # (B, C, H, W)
        
        # Save E for residual connection
        E_res = E
        
        # 4. Deep Feature Extraction (F_out)
        F_out = self.deep_feature_extractor(E)
        
        # 5. Feature Integration -> Residual Map R
        R = self.feature_integration(F_out) # (B, L_hsi, H, W)
        
        # 6. Final Fused Image (X = Z^ + R)
        hr_hsi = hsi_upsampled + R
        
        return {"pred": hr_hsi}
