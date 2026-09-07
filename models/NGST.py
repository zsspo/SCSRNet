import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from einops import rearrange
from timm.models.layers import  trunc_normal_, to_2tuple


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    wh, ww = H//window_size, W//window_size
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (wh, ww)


def window_unpartition(windows, num_windows):
    """
    Args:
        windows: [B*wh*ww, WH, WW, D]
        num_windows (tuple[int]): The height and width of the window.
    Returns:
        x: [B, ph, pw, D]
    """
    x = rearrange(windows, '(p h w) wh ww c -> p (h wh) (w ww) c', h=num_windows[0], w=num_windows[1])
    return x.contiguous()


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop):
        super().__init__()
        hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads,
            qkv_bias, attn_drop, proj_drop):
        super(WindowAttention, self).__init__()
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NGramContext(nn.Module):
    '''
    Args:
        dim (int): Number of input channels.
        window_size (int or tuple[int]): The height and width of the window.
        ngram (int): How much windows(or patches) to see.
        ngram_num_heads (int):
        padding_mode (str, optional): How to pad.  Default: seq_refl_win_pad
                                                   Options: ['seq_refl_win_pad', 'zero_pad']
    Inputs:
        x: [B, ph, pw D] or [B, C, H, W]
    Returns:
        context: [B, wh, ww, 1, 1, D] or [B, C, ph, pw]
    '''

    def __init__(self, dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad'):
        super(NGramContext, self).__init__()
        assert(padding_mode in ['seq_refl_win_pad', 'zero_pad'],
                "padding mode should be 'seq_refl_win_pad' or 'zero_pad'!!")

        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.ngram = ngram
        self.padding_mode = padding_mode

        self.unigram_embed = nn.Conv2d(dim, dim // 2,
                                       kernel_size=(self.window_size[0], self.window_size[1]),
                                       stride=self.window_size, padding=0, groups=dim // 2)
        self.ngram_attn = WindowAttention(dim=dim // 2, num_heads=ngram_num_heads,qkv_bias=False, attn_drop=0, proj_drop=0)
        self.avg_pool = nn.AvgPool2d(ngram)
        self.merge = nn.Conv2d(dim, dim, 1, 1, 0)

    def seq_refl_win_pad(self, x, back=False):
        if self.ngram == 1: return x
        x = TF.pad(x, (0, 0, self.ngram - 1, self.ngram - 1)) if not back else TF.pad(x, (self.ngram - 1, self.ngram - 1, 0, 0))
        if self.padding_mode == 'zero_pad':
            return x
        if not back:
            (start_h, start_w), (end_h, end_w) = to_2tuple(-2 * self.ngram + 1), to_2tuple(-self.ngram)
            # pad lower
            x[:, :, -(self.ngram - 1):, :] = x[:, :, start_h:end_h, :]
            # pad right
            x[:, :, :, -(self.ngram - 1):] = x[:, :, :, start_w:end_w]
        else:
            (start_h, start_w), (end_h, end_w) = to_2tuple(self.ngram), to_2tuple(2 * self.ngram - 1)
            # pad upper
            x[:, :, :self.ngram - 1, :] = x[:, :, start_h:end_h, :]
            # pad left
            x[:, :, :, :self.ngram - 1] = x[:, :, :, start_w:end_w]

        return x

    def sliding_window_attention(self, unigram):
        slide = unigram.unfold(3, self.ngram, 1).unfold(2, self.ngram, 1)
        slide = rearrange(slide, 'b c h w ww hh -> b (h hh) (w ww) c')  # [B, 2(wh+ngram-2), 2(ww+ngram-2), D/2]
        slide, num_windows = window_partition(slide, self.ngram)  # [B*wh*ww, ngram, ngram, D/2], (wh, ww)
        slide = slide.view(-1, self.ngram * self.ngram, self.dim // 2)  # [B*wh*ww, ngram*ngram, D/2]

        context = self.ngram_attn(slide).view(-1, self.ngram, self.ngram, self.dim // 2)  # [B*wh*ww, ngram, ngram, D/2]

        context = window_unpartition(context, num_windows)  # [B, wh*ngram, ww*ngram, D/2]
        context = rearrange(context, 'b h w d -> b d h w')  # [B, D/2, wh*ngram, ww*ngram]
        context = self.avg_pool(context)  # [B, D/2, wh, ww]
        return context

    def forward(self, x):
        B, ph, pw, D = x.size()
        x = rearrange(x, 'b ph pw d -> b d ph pw')  # [B, D, ph, pw]
        unigram = self.unigram_embed(x)  # [B, D/2, wh, ww]

        unigram_forward_pad = self.seq_refl_win_pad(unigram, False)  # [B, D/2, wh+ngram-1, ww+ngram-1]
        unigram_backward_pad = self.seq_refl_win_pad(unigram, True)  # [B, D/2, wh+ngram-1, ww+ngram-1]

        context_forward = self.sliding_window_attention(unigram_forward_pad)  # [B, D/2, wh, ww]
        context_backward = self.sliding_window_attention(unigram_backward_pad)  # [B, D/2, wh, ww]

        context_bidirect = torch.cat([context_forward, context_backward], dim=1)  # [B, D, wh, ww]
        context_bidirect = self.merge(context_bidirect)  # [B, D, wh, ww]
        context_bidirect = rearrange(context_bidirect, 'b d h w -> b h w d')  # [B, wh, ww, D]

        return context_bidirect.unsqueeze(-2).unsqueeze(-2).contiguous()  # [B, wh, ww, 1, 1, D]

class NGramWindowPartition(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        ngram (int): How much windows to see as context.
        ngram_num_heads (int):
        shift_size (int, optional): Shift size for SW-MSA.  Default: 0
    Inputs:
        x: [B, ph, pw, D]
    Returns:
        [B*wh*ww, WH, WW, D], (wh, ww)
    """

    def __init__(self, dim, window_size, ngram, ngram_num_heads, shift_size=0):
        super(NGramWindowPartition, self).__init__()
        self.window_size = window_size
        self.ngram = ngram
        self.shift_size = shift_size

        self.ngram_context = NGramContext(dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad')

    def forward(self, x):
        B, ph, pw, D = x.size()
        wh, ww = ph // self.window_size, pw // self.window_size  # number of windows (height, width)
        assert(0 not in [wh, ww], "feature map size should be larger than window size!")

        context = self.ngram_context(x)  # [B, wh, ww, 1, 1, D]

        windows = rearrange(x, 'b (h wh) (w ww) c -> b h w wh ww c',
                            wh=self.window_size,
                            ww=self.window_size).contiguous()  # [B, wh, ww, WH, WW, D]. semi window partitioning
        windows += context  # [B, wh, ww, WH, WW, D]. inject context

        # Cyclic Shift
        if self.shift_size > 0:
            x = rearrange(windows, 'b h w wh ww c -> b (h wh) (w ww) c').contiguous()  # [B, ph, pw, D]. re-patchfying
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))  # [B, ph, pw, D]. cyclic shift
            windows = rearrange(shifted_x, 'b (h wh) (w ww) c -> b h w wh ww c',
                                wh=self.window_size,
                                ww=self.window_size).contiguous()  # [B, wh, ww, WH, WW, D]. re-semi window partitioning
        windows = rearrange(windows,
                            'b h w wh ww c -> (b h w) wh ww c').contiguous()  # [B*wh*ww, WH, WW, D]. window partitioning

        return windows


class NGSTBlock(nn.Module):
    def __init__(self, dim,
                 num_heads,
                 window_size,
                 shift_size,
                 mlp_ratio,
                 qkv_bias,
                 drop,
                 attn_drop,
                 drop_path):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.ngram_window_partition = NGramWindowPartition(dim, window_size, 2, num_heads, shift_size=shift_size)   # N_Gram
        self.attn = WindowAttention(
            dim=dim, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = self.ngram_window_partition(x)  # [B*wh*ww, WH, WW, D], (wh, ww)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C


        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class NGST_Net(nn.Module):
    def __init__(self,
                 config,
                 dim=48,
                 num_heads=3,
                 window_size=8,
                 mlp_ratio=1,
                 qkv_bias =False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 depths=6):
        super().__init__()
        
        inp_channels = config[config["train_dataset"]]["spectral_bands"]

        self.HSI_bands = config[config["train_dataset"]]["spectral_bands"]
        self.scale_ratio = config[config["train_dataset"]]["factor"]
        self.lr_size =  config[config["train_dataset"]]["LR_size"]
        self.hr_size =  config[config["train_dataset"]]["HR_size"]
        self.MSI_bands = config[config['train_dataset']]["msi_bands"]
        self.selected_sp_channels = [config[config['train_dataset']]['B'], config[config['train_dataset']]['G'], config[config['train_dataset']]['R']]
    
        self.dim = dim

        drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        self.Embedding = nn.Sequential(
            nn.Linear(self.HSI_bands + self.MSI_bands, dim),
        )
        self.blocks = nn.ModuleList([
            NGSTBlock(dim=dim,
                      num_heads=num_heads, window_size=window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      drop=drop_rate, attn_drop=attn_drop_rate,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                      )
            for i in range(depths)])

        self.refine = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.dim, self.HSI_bands, 3, 1, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, lrhsi, hrmsi):
        up_lrhsi = F.interpolate(lrhsi, scale_factor=self.scale_ratio, mode='bilinear')  # (b N h w)
        up_lrhsi = up_lrhsi.clamp_(0, 1)
        x = torch.cat((up_lrhsi , hrmsi),1)   #Data cube
        x_size = (x.shape[2], x.shape[3])

        x = rearrange(x, 'B c H W -> B (H W) c', H=x_size[0])
        x = self.Embedding(x)

        for blk in self.blocks:
            x = blk(x, x_size)

        x= rearrange(x, 'B (H W) C -> B C H W', H=x_size[0])
        x = self.refine(x)

        out = x + up_lrhsi
        out = out.clamp_(0, 1)

        return {'pred': out}



# model = NGST_Net(scale_ratio = 4,
#                  MSI_bands = 3,
#                  HSI_bands = 128,)
#
# hrmsi = torch.randn((1, 3, 64, 64))
# lrhsi = torch.randn((1, 128, 16, 16))
#
# x = model(lrhsi, hrmsi)
# print(x.shape)