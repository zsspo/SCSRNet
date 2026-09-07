import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import torch.utils.checkpoint as checkpoint
from functools import partial
#from ..src.shiftadd.ops import AddShift_ops
from ops import AddShift_ops
#from lib.channel_position import ChannelWeightedBlock
#from lib.lib import *
import numpy as np
import math

class GRNwithNHWC(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


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

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)
    
class SSEBlock(nn.Module):
    """
    """
    def __init__(self, input_channels, internal_neurons):
        super(SSEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs, y):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return y * x.view(-1, self.input_channels, 1, 1)

    
class ECABlock(nn.Module):
    """ECA module"""
    def __init__(self, channels, b=1, gamma=2):
        super(ECABlock,self).__init__()
        #自适应卷积核大小
        self.kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        if self.kernel_size % 2 ==0 :
            self.kernel_size = self.kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1)//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: 生成通道权重的特征
        y: 用于注意力的特征,
        同一尺度下，将融合后的特征通道对齐到
        """
        b, c, h, w = x.size()
        z = self.avg_pool(x)

        # Two different branches of ECA module
        z = self.conv(z.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        z = self.sigmoid(z)
        return x*z.expand_as(x)  #维度扩展

class LayerNorm(nn.Module):
    """ 
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    
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
    #融合相邻的卷积和bn层
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()  #计算标准差
    #bn.weigh,bn.bias分别为bn层可学习缩放操作的参数gamma，beta
    #runnning_mean,running_var分别为训练阶段统计的均值方差，测试用到
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    #分支合并
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)  #形状为bch单位矩阵
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
    large_k = large_kernel.size(2)                              #期望得到的大核尺寸
    dilated_k = dilated_kernel.size(2)                          #空洞卷积的核尺寸
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1    #空洞卷积扩张后的核尺寸
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r) #空洞卷积转换为对应的为大核卷积
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2     #将大核卷积补齐至期望的大核尺寸
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class DilatedBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)    #定义多分支卷积块的实际核，训练时无bias，后面跟original_bn,部署时有bias
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 31:
            self.kernel_sizes = [5, 9, 11, 5, 5, 3, 3, 3] #5, 17, 31, 21, 25, 15, 17, 19
            self.dilates = [1, 2, 3, 5, 6, 7, 8, 9]
        elif kernel_size == 19:
            self.kernel_sizes = [5, 9, 7, 3, 3] #5, 17, 19, 11, 15
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 9, 5, 3, 3] #5, 17, 13, 11, 
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 15: 
            self.kernel_sizes = [5, 7, 3, 3, 3, 3] #5, 13, 7, 9, 11, 15
            self.dilates = [1, 2, 3, 4, 5, 7]
        elif kernel_size == 13:
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
            self.origin_bn = get_bn(channels, use_sync_bn)  #为每个多分支卷积块加上属性字段
            for k, r in zip(self.kernel_sizes, self.dilates):  #遍历所有分支，添加对应核尺寸和扩张率的卷积层、bn层
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))  #这里空洞卷积还是原始尺寸加上空洞率实现的
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy modebranch_k
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):    #遍历扩张卷积分支
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))   #所有分支的输出相加，包括原始大核分支
        return out

    def merge_dilated_branches(self):
        #在训练完后进行推理时合并
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)   #先融合原始大核的卷积和bn层
            for k, r in zip(self.kernel_sizes, self.dilates):              #遍历所有扩张卷积分支，合并到原始大核卷积层上
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)  #新定义一个大核卷积用来承接合并后的卷积权重和偏置
            merged_conv.weight.data = origin_k  
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))
                
  
class LKSpatiaSpectalFusionBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super(LKSpatiaSpectalFusionBlock, self).__init__()
        
        
    def forward(self, ):
        
        return 
        
        
class LoRAConvsByRandom(nn.Module):
    '''
    merge LoRA1 LoRA2 small_conv
    random set index for three branch
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 big_kernel, small_kernel,
                 stride=1, group=1,
                 bn=True, use_small_conv=True):
        super().__init__()
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride
        self.small_conv = use_small_conv
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        # self.split_convs = nn.Conv2d(in_channels, out_n,
        #                              kernel_size=small_kernel, stride=stride,
        #                              padding=padding, groups=group,
        #                              bias=False)
        # only part of input will using shift-wise
        ghost_ratio=0.3
        ghostN=int(in_channels*ghost_ratio)
        repN= in_channels - ghostN
        np.random.seed(123)
        ghost = np.random.choice(in_channels,ghostN,replace=False).tolist()
        ghost.sort()
        rep=list(set(range(in_channels))-set(ghost))
        rep.sort()
        assert len(rep)==repN,f'len(rep):{len(rep)}==repN:{repN}'
        self.ghost=torch.IntTensor(ghost)
        self.rep=torch.IntTensor(rep)
        # self.ghost=torch.arange(ghostN).int()
        # self.rep=torch.arange(ghostN, in_channels).int()

        N_rep=4
        out_n = repN * self.nk
        self.split_convs = nn.ModuleList([
            nn.Conv2d(repN, out_n,
                      kernel_size=small_kernel, stride=stride,
                      padding=padding, groups=repN,
                      bias=False)
            for _ in range(N_rep)
        ])
        self.device = None
        self.lora1 = None
        print(f'ghost_ratio={ghost_ratio},N_rep={N_rep}')
        torch.manual_seed(123)
        self.lora2 = torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(repN)])# shuffle in group
        # self.lora2 = torch.cat([torch.arange(self.nk-1,-1,-1).int() + i * self.nk for i in range(repN)])# shuffle in group
        # np.random.seed(123) 
        # self.small = torch.IntTensor(np.random.randint(0,out_n,out_channels)) if use_small_conv else None
        torch.manual_seed(123)
        self.small = (torch.randint(0,self.nk,[repN])+torch.arange(repN)*self.nk).int() if use_small_conv else None
        self.use_bn = bn
        
        self.loras=[lora_module(padding, real_pad, small_kernel,self.lora2, repN, out_n)
                    for _ in range(N_rep)]

        if bn:
            self.bn_lora1 = get_bn(repN)
            self.bn_lora2 = get_bn(repN)
            self.bn_lora = get_bn(repN)
            # self.bn_small = get_bn(out_channels) if use_small_conv else None
            # self.bn_lora1 = nn.ModuleList([get_bn(out_channels) for _ in range(N_rep)])
            # self.bn_lora2 = nn.ModuleList([get_bn(out_channels) for _ in range(N_rep)])
        else:
            self.bn_lora1 = None
            self.bn_lora2 = None
            self.bn_lora = None
            # self.bn_lora1 = [None]#*N_rep
            # self.bn_lora2 = [None]*N_rep
        if bn and use_small_conv:
            self.bn_small = get_bn(repN)
            # self.bn_small = nn.ModuleList([get_bn(out_channels) for _ in range(N_rep)])
        else:
            self.bn_small = None
            # self.bn_small = [None]*N_rep
        self.inputsize=True


    def forward(self, inputs):
        if self.inputsize:
            print(f'shape:{inputs.shape}; kernel:{self.kernels}')
            self.inputsize=False
            
        # split output
        ori_b, ori_c, ori_h, ori_w = inputs.shape
        if self.device is None:
            self.device = inputs.get_device()
            if self.device ==-1:#cpu
                self.device=None
            else:
                self.lora2 = self.lora2.to(self.device)
                self.small = self.small.to(self.device) if self.small_conv else None
                self.ghost=self.ghost.to(self.device)
                self.rep=self.rep.to(self.device)
        ghost_inputs = torch.index_select(inputs, 1, self.ghost)
        rep_inputs = torch.index_select(inputs, 1, self.rep)
        lora1_x = 0
        lora2_x = 0
        small_x = 0
        # for (split_convs, bn_lora1, bn_lora2, bn_small) in zip(self.split_convs, self.bn_lora1, self.bn_lora2, self.bn_small):
        outs=[]
        for split_convs in self.split_convs :
        # for split_convs, lora in zip(self.split_convs, self.loras):
            out = split_convs(rep_inputs)
            # outs.append(out)
            # l1x = self.forward_lora(out, ori_h, ori_w)#, bn=bn_lora1)
            # l2x = self.forward_lora(out, ori_h, ori_w, idx=self.lora2, VH='W')#, bn=bn_lora2)
            # o1x,o2x=lora(out, ori_b, ori_h, ori_w) # torch.abs(l1x-o1x).sum(), torch.abs(l2x-o2x).sum()
            lora1_x += self.forward_lora(out, ori_h, ori_w)#, bn=bn_lora1)
            lora2_x += self.forward_lora(out, ori_h, ori_w, idx=self.lora2, VH='W')#, bn=bn_lora2)
            if self.small_conv:
                small_x += self.forward_small(out, self.small)#, bn_small
        if self.use_bn:
            lora1_x = self.bn_lora1(lora1_x)
            lora2_x = self.bn_lora2(lora2_x) 
            small_x = self.bn_small(small_x) 
        x = lora1_x + lora2_x + small_x
        if self.use_bn:
            x = self.bn_lora(x) 
        x=torch.cat([x,ghost_inputs],dim=1)
        # torch.save(outs,'se.pt')
        return x
        #     # out_rep = self.split_rep_convs(inputs)
        #         # small_rep_x = self.forward_small(out_rep, self.small, self.bn_rep_small)
        #         # x_rep += small_rep_x
        # # y=x+x_rep
        #     # x_rep = lora1_rep_x + lora2_rep_x
        #     # lora1_rep_x = self.forward_lora(out_rep, ori_h, ori_w, bn=self.bn_rep_lora1)
        #     # lora2_rep_x = self.forward_lora(out_rep, ori_h, ori_w, VH='W', idx=self.lora2, bn=self.bn_rep_lora2)

    def forward_small(self, out, idx): #, small_bn):
        # shift along the index of every group
        b, c, h, w = out.shape
        out = torch.index_select(out, 1, idx)
        padding, *_ = self.pad
        k = min(self.kernels) 
        pad = padding - k // 2
        if pad>0:
            out = torch.narrow(out, 2, pad, h - 2 * pad)
            out = torch.narrow(out, 3, pad, w - 2 * pad)
        # if self.use_bn:
        #     out = small_bn(out)
        return out

    def forward_lora(self, out, ori_h, ori_w, VH='H', idx=None, bn=None):
        # shift along the index of every group
        b, c, h, w = out.shape
        if idx is not None:
            # out=out[idx]
            out = torch.index_select(out, 1, idx)
        out = torch.split(out.reshape(b, -1, self.nk, h, w), 1, 2)  # ※※※※※※※※※※※
        x = 0
        for i in range(self.nk):
            outi = self.rearrange_data(out[i], i, ori_h, ori_w, VH)
            x = x + outi
        # if self.use_bn:
        #     x = bn(x)
        return x

    def rearrange_data(self, x, idx, ori_h, ori_w, VH):
        padding, pads = self.pad
        x = x.squeeze(2)  # ※※※※※※※
        *_, h, w = x.shape
        k = min(self.kernels)
        pad=pads[idx] 

        ori_k = max(self.kernels)
        ori_p = ori_k // 2
        stride = self.stride
        # need to calculate start point after conv
        # how many windows shift from real start window index
        if pad<0:
            pad_l = 0
            s = 0-pad
        else:
            pad_l = pad
            s = 0
        if VH == 'H':
            # assume add sufficient padding for origin conv
            suppose_len = (ori_w + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (w + pad_l) else s + suppose_len - w - pad_l
            new_pad = (pad_l, pad_r, 0, 0)
            dim = 3
            e = w + pad_l + pad_r - s - suppose_len
        else:
            # assume add sufficient padding for origin conv
            suppose_len = (ori_h + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (h + pad_l) else s + suppose_len - h - pad_l
            new_pad = (0, 0, pad_l, pad_r)
            dim = 2
            e = h + pad_l + pad_r - s - suppose_len
        # print('new_pad', new_pad)
        if len(set(new_pad)) > 1:
            x = F.pad(x, new_pad)
         
        # padding on other direction
        # if padding * 2 + 1 != k:
        #     pad = padding - k // 2
        yy = padding - k // 2
        if yy>0:
            if VH == 'H':  # horizonal
                # x = torch.narrow(x, 2, pad, h - 2 * pad)
                x = torch.narrow(x, 2, yy, h - 2 * yy)
            else:  # vertical
                # x = torch.narrow(x, 3, pad, w - 2 * pad)
                x = torch.narrow(x, 3, yy, w - 2 * yy)

        xs = torch.narrow(x, dim, s, suppose_len)
        return xs

    def shift(self, kernels):
        '''
        We assume the conv does not change the feature map size, so padding = bigger_kernel_size//2. Otherwise,
        you may configure padding as you wish, and change the padding of small_conv accordingly.
        '''
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink) 
        # 2. padding
        padding = mink -1  
        # padding = mink // 2
        # 3. pads for each idx
        mid=maxk // 2
        real_pad=[]
        for i in range(nk): 
            extra_pad=mid-i*mink - padding 
            real_pad.append(extra_pad)
        return padding, real_pad


class LKSABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(LKSABlock, self).__init__()
        #self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.Sigmoid())
        self.attention = nn.Sequential(DilatedBlock(in_channels, 7, deploy=False,use_sync_bn=False,
                                              attempt_use_lk_impl=True), nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x, y):
        """
        x: 输入特征以得到权重
        y: 用于注意力的特征
        """
        attention_mask = self.attention(x)
        features = self.conv(y)
        return torch.mul(features, attention_mask)


class SSCBlock(nn.Module):
    """ spectral and spatial consistency. SA and CA for fusion of c&s """
    def __init__(self, in_channels, out_channels):
        super(SSCBlock, self).__init__()
        #输入输出通道为融合特征的通道
        self.in_channels = in_channels   #2c
        self.out_channels = out_channels  #c

        self.CA = SSEBlock(self.in_channels, self.in_channels//4)
        self.SA = LKSABlock(self.in_channels, self.in_channels)
        self.fuse = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, hsi_feats, msi_feats, feats):
        #fused_feats = self.FE(fused_feats)        #
        sa = self.SA(msi_feats, feats)
        ca = self.CA(hsi_feats, feats)
        feats = sa + ca  
        feats = self.fuse(feats)     
        return feats

class LKSSBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 decom=True,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        self.decom = decom
        self.deploy = deploy
        if deploy:
            print('------------------------------- Note: deploy mode')
        if self.with_cp:
            print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        if kernel_size == 0:
            self.spatial_dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.spatial_dwconv = DilatedBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)

        else:
            assert kernel_size in [3, 5]
            self.spatial_dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=deploy,
                                     attempt_use_lk_impl=attempt_use_lk_impl)

        
        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.weighted_spec = ECABlock(dim)

        ffn_dim = int(ffn_factor * dim)
        self.spectral_pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        #self.gelu = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim, use_bias=not deploy))
        self.gelu = nn.GELU()
        if deploy:
            self.spectral_pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.spectral_pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.alpha = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

    def compute_residual(self, x):
        y = self.weighted_spec(self.norm(self.spatial_dwconv(x)))
        y = self.spectral_pwconv2(self.gelu(self.spectral_pwconv1(y))) 
        if self.alpha is not None:
            y = self.alpha.view(1, -1, 1, 1) * y  #残差缩放
        return self.drop_path(y)


    def forward(self, inputs):

        def _f(x):
            x = x + self.compute_residual(x)
            return x 
        
        if self.with_cp and inputs.requires_grad:
            out = checkpoint.checkpoint(_f, inputs)
        else:
            out = _f(inputs)
        return out

    #重参数化部分
    def reparameterize(self):
        if hasattr(self.spatial_dwconv, 'merge_dilated_branches'):
            self.spatial_dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.spatial_dwconv, 'lk_origin'):
                self.spatial_dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.spatial_dwconv.lk_origin.bias.data = self.norm.bias + (
                            self.spatial_dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.spatial_dwconv.in_channels, self.spatial_dwconv.out_channels, self.spatial_dwconv.kernel_size,
                                 padding=self.spatial_dwconv.padding, groups=self.spatial_dwconv.groups, bias=True)
                conv.weight.data = self.spatial_dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()
        if self.alpha is not None:
            final_scale = self.alpha.data
            self.alpha = None
        else:
            final_scale = 1
        if self.gelu[1].use_bias and len(self.spectral_pwconv2) == 3:
            grn_bias = self.gelu[1].beta.data
            self.gelu[1].__delattr__('beta')
            self.gelu[1].use_bias = False
            linear = self.spectral_pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.spectral_pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.spectral_pwconv2 = nn.Sequential(new_linear, self.spectral_pwconv2[1])


default_kernel_sizes = ((3, 13),
                        (31, 3),
                        (19, 3),
                        (17, 3))

default_kernel_sizes1 = ((3, 3),
                        (13, 13),
                        (13, 13, 13, 13),
                        (13, 13))
default_kernel_sizes2 = ((3, 3),
                        (13, 13),
                        (13, 13, 13, 13, 13, 13, 13, 13),
                        (13, 13))
default_kernel_sizes3 = ((3, 3, 3),
                        (13, 13, 13),
                        (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
                        (13, 13, 13))
default_kernel_sizes4 = ((3, 3, 3),
                        (13, 13, 13),
                        (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
                        (13, 13, 13))


depths = (2, 2, 2, 2)
depths1 = (2, 2, 8, 2)
depths2 = (3, 3, 18, 3)
depths3 = (3, 3, 27, 3)

default_depths_to_kernel_sizes = {
    depths: default_kernel_sizes,
    depths1: default_kernel_sizes1,
    depths2: default_kernel_sizes2,
    depths3: default_kernel_sizes3
}


class HyperLKNet(nn.Module):
    """ HyperLKNet
        A PyTorch impl of HyperLKNet

    Args:
        in_chans (int): Number of input channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 27, 3)
        dims (int): Feature dimension at each stage. Default: (96, 192, 384, 768)
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        kernel_sizes (tuple(tuple(int))): Kernel size for each block. None means using the default settings. Default: None.
        deploy (bool): deploy = True means using the inference structure. Default: False
        with_cp (bool): with_cp = True means using torch.utils.checkpoint to save GPU memory. Default: False
        init_cfg (dict): weights to load. The easiest way to use UniRepLKNet with for OpenMMLab family. Default: None
        attempt_use_lk_impl (bool): try to load the efficient iGEMM large-kernel impl. Setting it to False disabling the iGEMM impl. Default: True
        use_sync_bn (bool): use_sync_bn = True means using sync BN. Use it if your batch size is small. Default: False
    """
    def __init__(self,
                 config,
                 depths=depths,
                 dims=(64, 64, 64, 64),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 kernel_sizes=None,
                 deploy=False,
                 with_cp=False,
                 init_cfg=None,
                 attempt_use_lk_impl=True,
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

        #hsi_wavelength_range = config[config["train_dataset"]]["hsi_wavelength_range"]
        #msi_wavelength_range = config[config["train_dataset"]]["msi_wavelength_range"]
        #hsi_wavelength_step = (hsi_wavelength_range[1]- hsi_wavelength_range[0])/(self.in_channels-1)
        #msi_wavelength_step = (msi_wavelength_range[1]- msi_wavelength_range[0])/(self.msi_channels-1)
        #self.hsi_wavelengths = torch.from_numpy(np.arange(hsi_wavelength_range[0], hsi_wavelength_range[1]+1, hsi_wavelength_step, dtype=np.float32))
        #self.msi_wavelengths = torch.from_numpy(np.arange(msi_wavelength_range[0], msi_wavelength_range[1]+1, msi_wavelength_step, dtype=np.float32)) 
        
    
        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                print('=========== use default kernel size ')
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError('no default kernel size settings for the given depths')
        print(kernel_sizes)
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i], 'kernel sizes do not match the depths'

        self.with_cp = with_cp

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        print('=========== drop path rates: ', dp_rates)
        
        #blocks
        self.blocks = nn.ModuleList()
        cur = 0
        for i in range(4):
            block = nn.Sequential(
                *[LKSSBlock(dim=dims[i], kernel_size=kernel_sizes[i][j], drop_path=dp_rates[cur + j],
                                   layer_scale_init_value=layer_scale_init_value, deploy=deploy,
                                   attempt_use_lk_impl=attempt_use_lk_impl,
                                   with_cp=with_cp, use_sync_bn=use_sync_bn) for j in range(depths[i])])
            self.blocks.append(block)
            cur += depths[i]
            
        first_channels = dims[0]
        last_channels = dims[-1]
        
        
        self.hsi_layer = nn.Conv2d(self.in_channels, first_channels, kernel_size=3, stride=1, padding=1)
        self.msi_layer = nn.Conv2d(self.msi_channels, first_channels, kernel_size=3, stride=1, padding=1)
        self.input_layer = nn.Conv2d(first_channels*2, first_channels, kernel_size=3, stride=1, padding=1)
        self.refine = SSCBlock(last_channels, self.out_channels)
        
        self.for_pretrain = init_cfg is None
        self.for_downstream = not self.for_pretrain     # there may be some other scenarios
     
        self.init_cfg = init_cfg        
        self.init_weights()
        
    def forward(self, lr_hsi, hr_msi):  
        #channel ware
        hsi_up = F.interpolate(lr_hsi, scale_factor=self.factor, mode='bicubic') 
        
        x_hsi = self.hsi_layer(hsi_up)
        x_msi = self.msi_layer(hr_msi)

        #空间和波长编码
        x = torch.cat((x_hsi, x_msi), dim=1) #
        #x = torch.cat((hsi_up, hr_msi), dim=1)
        x = self.input_layer(x)
      
        for stage_idx in range(4):
            #x = self.downsample_layers[stage_idx](x)
            x = self.blocks[stage_idx](x)
            #outs.append(self.__getattr__(f'norm{stage_idx}')(x))
        x = self.refine(x_hsi, x_msi, x) + hsi_up
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

            
if __name__ == '__main__':
    #   Test case showing the equivalency of Structural Re-parameterization
    x = torch.randn(2, 4, 19, 19).cuda()
    #model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(32, 64, 128, 256), **kwargs)
    layer = LKSSBlock(4, kernel_size=13, small_kernel=5, attempt_use_lk_impl=True).cuda()
    for n, p in layer.named_parameters():
        if 'beta' in n:
            torch.nn.init.ones_(p)
        else:
            torch.nn.init.normal_(p)
    for n, p in layer.named_buffers():
        if 'running_var' in n:
            print('random init var')
            torch.nn.init.uniform_(p)
            p.data += 2
        elif 'running_mean' in n:
            print('random init mean')
            torch.nn.init.uniform_(p)
    layer.gamma.data += 0.5
    
    layer.eval()
    origin_y = layer(x)
    layer.reparameterize()
    eq_y = layer(x)
    print(layer)
    print(eq_y - origin_y)
    print((eq_y - origin_y).abs().sum() / origin_y.abs().sum())
