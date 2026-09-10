import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import torch.utils.checkpoint as checkpoint
from functools import partial
import numpy as np
import math

class NCHWtoNHWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)

    
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)

    if attempt_use_lk_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)
    

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))

    if bn:
        result.add_module('bn', get_bn(out_channels))
    return result

def fuse_bn(conv, bn):
    # Fuse adjacent conv and bn layers
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()  # Compute standard deviation
    # bn.weight, bn.bias are the learnable scaling parameters gamma, beta of the bn layer
    # running_mean, running_var are the mean and variance statistics collected during training, used in inference
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    # Branch merging
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)  # Identity matrix with bch shape
    if kernel.size(1) == 1:  #
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)                              # Expected large kernel size
    dilated_k = dilated_kernel.size(2)                          # Kernel size of the dilated convolution
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1    # Kernel size after dilated convolution expansion
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r) # Convert dilated convolution to equivalent large-kernel convolution
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2     # Pad the large kernel to the expected size
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel

class MSPM(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl) 

        #  Default settings
        if kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3] #5, 13, 7, 9, 11
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Undefined kernel sizes!')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)  # Add attribute field for each multi-branch conv block
            for k, r in zip(self.kernel_sizes, self.dilates):  # Iterate over all branches, add conv and bn layers with corresponding kernel sizes and dilation rates
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))  # Here the dilated convolution is still implemented with original size plus dilation rate
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):    # Iterate over dilation conv branches
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))   # Sum the outputs of all branches, including the original large-kernel branch
        return out

    def merge_dilated_branches(self):
        # Merge branches during inference after training
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)   # First fuse the conv and bn of the original large-kernel branch
            for k, r in zip(self.kernel_sizes, self.dilates):              # Iterate over all dilation conv branches, merge into the original large-kernel conv
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)  # Define a new large-kernel conv to hold the merged weights and bias
            merged_conv.weight.data = origin_k  
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

class DynamicFusionGate(nn.Module):
    """
    Dynamic Spectral-Spatial Fusion Gate
    """
    def __init__(self, channels, reduction=4):
        super(DynamicFusionGate, self).__init__()
        hidden_dim = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.refine = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )

    def forward(self, x_hsi, x_msi):
        fusion_input = torch.cat([x_hsi, x_msi], dim=1)
        weight = self.gate(fusion_input)
        fused = weight * x_hsi + (1 - weight) * x_msi
        fused = self.refine(fused)
        return fused

class ChannelConvBranch(nn.Module):
    """1D convolution along the channel dimension (along C), each spatial location is independent.
    Input: (B, C, H, W)
    Output: (B, C, H, W)   # Keep the same shape
    """
    def __init__(self, channels, kernel_size, activation=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        padding = kernel_size // 2   # Keep channel count unchanged

        # Conv kernel: (out_channels=1, in_channels=1, kernel=(k, 1))
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            bias=False
        )
        if activation:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        # (B, C, H, W) -> (B, 1, C, L)
        x_reshaped = x.view(B, C, L).unsqueeze(1)   # (B, 1, C, L)
        # 1D convolution along the channel (C) dimension
        out = self.conv(x_reshaped)                 # (B, 1, C, L)
        out = self.act(out)
        # Restore shape (B, C, H, W)
        out = out.squeeze(1).view(B, C, H, W)
        return out


class SpatialGuidedSpectralRouting(nn.Module):
    def __init__(
            self,
            channels,
            branch_configs=((3, True), (5, True), (7, True)),
            reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.routing = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, len(branch_configs), 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.spectral_proj = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False
        )
        self.branches = nn.ModuleList([ChannelConvBranch(
            channels=channels,
            kernel_size=k,
            activation=act
            ) for k, act in branch_configs])

    def forward(self, x):
        route = self.routing(x)
        # B,n_branch,H,W
        route = self.softmax(route)
        s = self.spectral_proj(x)
        branch_outputs = [branch(s) for branch in self.branches]
        out = s
        for i, b in enumerate(branch_outputs):
            out = out + route[:, i:i+1,:,:] * b
        return out

class LKSSBlock(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 scale_value=1e-6,
                 deploy=False,
                 decom=True,
                 attempt_use_lk_impl=False,
                 with_cp=False,
                 use_sync_bn=False,
                 ):
        super().__init__()
        self.with_cp = with_cp
        self.decom = decom
        self.deploy = deploy

        if deploy:
            print('------------------------------- Note: deploy mode')
        if self.with_cp:
            print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        if kernel_size == 0:
            self.spa = nn.Identity()
        elif kernel_size >= 7: # LKSSRB
            self.spa = MSPM(dim, kernel_size, deploy=deploy,
                                            use_sync_bn=use_sync_bn,
                                            attempt_use_lk_impl=attempt_use_lk_impl)

        else:
            assert kernel_size in [3, 5] # SKSSRB
            self.spa = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=deploy,
                                     attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.spec = SpatialGuidedSpectralRouting(dim)
        self.gamma = nn.Parameter(scale_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and scale_value is not None \
                                                         and scale_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def residual(self, x):
        y = self.norm(self.spa(x))
        y = self.spec(y)
        y = self.gamma.view(1, -1, 1, 1) * y  # Residual scaling
        return self.drop_path(y)

    def forward(self, inputs):
        return inputs + self.residual(inputs)

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                            self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 padding=self.dwconv.padding, groups=self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()



default_kernel_sizes = ((3, 3),
                        (13, 13),
                        )

depths = (2, 2)
default_depths_to_kernel_sizes = {
    depths: default_kernel_sizes,
}


class SCSRNet(nn.Module):
    """A PyTorch impl of SCSRNet"""
    def __init__(self,
                 config,
                 depths=depths,
                 dims=(64, 64),
                 drop_path_rate=0.,
                 scale_value=1e-6,
                 kernel_sizes=None,
                 deploy=False,
                 with_cp=False,
                 init_cfg=None,
                 attempt_use_lk_impl=False,
                 use_sync_bn=False,
                 **kwargs
                 ):
        super().__init__()

        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.msi_channels = config[config["train_dataset"]]["msi_bands"]
        self.factor = config[config["train_dataset"]]["factor"]
        self.lr_size =  config[config["train_dataset"]]["LR_size"]
        self.hr_size =  config[config["train_dataset"]]["HR_size"]
        self.n_select_bands = config[config['train_dataset']]["msi_bands"]
        self.selected_sp_channels = [config[config['train_dataset']]['B'], config[config['train_dataset']]['G'], config[config['train_dataset']]['R']]
        #scale_value = config['scale_f']

        depths = tuple(depths)
        self.num_layers = len(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                print('=========== use default kernel size ')
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError('no default kernel size settings for the given depths')
        print(kernel_sizes)
        for i in range(self.num_layers):
            assert len(kernel_sizes[i]) == depths[i], 'kernel sizes do not match the depths'

        self.with_cp = with_cp

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        print('=========== drop path rates: ', dp_rates)
        
        self.blocks = nn.ModuleList()
        cur = 0
        for i in range(self.num_layers):
            block = nn.Sequential(
                *[LKSSBlock(dim=dims[i], kernel_size=kernel_sizes[i][j], drop_path=dp_rates[cur + j],
                                   scale_value=scale_value, deploy=deploy,
                                   attempt_use_lk_impl=attempt_use_lk_impl,
                                   with_cp=with_cp, use_sync_bn=use_sync_bn) for j in range(depths[i])])
            self.blocks.append(block)
            cur += depths[i]
            
        first_channels = dims[0]
        last_channels = dims[-1]
        
        self.hsi_layer = nn.Conv2d(self.in_channels, first_channels, kernel_size=3, stride=1, padding=1)
        self.msi_layer = nn.Conv2d(self.msi_channels, first_channels, kernel_size=3, stride=1, padding=1)

        self.gate = DynamicFusionGate(first_channels)
        # self.input_layer = nn.Conv2d(first_channels*2, first_channels, kernel_size=3, stride=1, padding=1)

        self.out = nn.Conv2d(last_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.embedding_dim = 64
        self.for_pretrain = init_cfg is None
    
        self.init_cfg = init_cfg        
        self.init_weights()
        
    def forward(self, lr_hsi, hr_msi):  
        # channel wise
        hsi_up = F.interpolate(lr_hsi, scale_factor=self.factor, mode='bicubic') 
        B, C, H, W = hsi_up.shape
    
        x_hsi = self.hsi_layer(hsi_up)
        x_msi = self.msi_layer(hr_msi)
        x = self.gate(x_hsi, x_msi)

        # SKSSRBs and LKSSRBs
        for stage_idx in range(self.num_layers):
            x = self.blocks[stage_idx](x)

        x = self.out(x) + hsi_up
        return {"pred": x}
    
    def reparameterize_hyperlkn(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()
                
    #   load pretrained model weights in the OpenMMLab style
    def init_weights(self):
        def load_state_dict(module, state_dict, strict=False, logger=None):
            unexpected_keys = []
            own_state = module.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    unexpected_keys.append(name)
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            missing_keys = set(own_state.keys()) - set(state_dict.keys())

            err_msg = []
            if unexpected_keys:
                err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
            if missing_keys:
                err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
            err_msg = '\n'.join(err_msg)
            if err_msg:
                if strict:
                    raise RuntimeError(err_msg)
                elif logger is not None:
                    logger.warn(err_msg)
                else:
                    print(err_msg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)




