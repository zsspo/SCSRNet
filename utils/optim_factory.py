# UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition
# Github source: https://github.com/AILab-CVC/UniRepLKNet
# Licensed under The Apache License 2.0 License [see LICENSE for details]
# Based on RepLKNet, ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/DingXiaoH/RepLKNet-pytorch
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import NAdamLegacy
from timm.optim.radam import RAdamLegacy
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_convnext(var_name):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3 
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_convnext(var_name)


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # if weight_decay and filter_bias_and_bn:
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = NAdamLegacy(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdamLegacy(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


from torch.optim.lr_scheduler import ReduceLROnPlateau


class LRWarmupScheduler:
    """This class wraps the standard PyTorch LR scheduler to support warmup."""

    def __init__(self, torch_scheduler, by_epoch, epoch_len=None,
                 # 以下为warmup相关设置
                 warmup_t=0, warmup_by_epoch=False, warmup_mode="fix",
                 warmup_init_lr=None, warmup_factor=None):
        # PyTorch原生的lr scheduler
        self.torch_scheduler = torch_scheduler
        # 表示torch_scheduler是by epoch还是by iter
        self.by_epoch = by_epoch
        # 每个epoch的长度，只有by_epoch=True且warmup_by_epoch=False时才需要传入这个参数
        self.epoch_len = epoch_len
        # 后面有很多代码需要同时处理by epoch和by iter的情况，将变量命名为epoch或iter都不合适，
        # 所以选择用t来代表这层含义。当warmup_by_epoch=True时，warmup_t代表warmup_epochs，
        # 反之代表warmup_iters。
        self.warmup_t = warmup_t
        # 表示warmup是by epoch还是by iter
        self.warmup_by_epoch = warmup_by_epoch
        # 取值为fix、auto、factor
        self.warmup_mode = warmup_mode
        # fix模式的warmup初始学习率
        self.warmup_init_lr = warmup_init_lr
        # auto和factor模式下，warmup的初始学习率为base_lr * warmup_factor
        self.warmup_factor = warmup_factor

        self.param_groups = self.torch_scheduler.optimizer.param_groups
        self.base_lrs = [param_group["lr"] for param_group in self.param_groups]
        # 因为factor模式需要知道常规学习率才能推导出warmup学习率，所以需要预先计算出torch_scheduler在每个t的常规学习率。
        # 假设by_epoch=True & warmup_by_epoch=False & warmup_t=25 & epoch_len=10，
        # 说明warmup阶段跨越了3个epoch，我们需要预先计算出torch_scheduler在前三个epoch的
        # 常规学习率（保存在self.regular_lrs_per_t中）。
        # PS：虽然很多PyTorch原生的lr scheduler（StepLR、MultiStepLR、CosineAnnealingLR）
        # 提供了学习率的封闭形式，即_get_closed_form_lr()函数，可以通过传入的epoch参数直接计算出对应的学习率。
        # 但仍有些scheduler并未提供此功能，例如CosineAnnealingWarmRestarts。
        # 所以这里只能通过step()函数来一步步地计算出学习率。
        max_t = warmup_t // epoch_len if by_epoch and not warmup_by_epoch else warmup_t
        self.regular_lrs_per_t = self._pre_compute_regular_lrs_per_t(max_t)

        self.last_iter = self.last_epoch = 0
        self.in_iter_warmup = False

        if warmup_by_epoch:
            assert by_epoch
        if by_epoch and not warmup_by_epoch:
            assert epoch_len is not None
        if self._is_plateau:
            assert by_epoch
        if warmup_t > 0:
            if warmup_mode == "fix":
                assert isinstance(warmup_init_lr, float)
                # 为第0个t准备好学习率
                self._set_lrs(warmup_init_lr)
            elif warmup_mode == "factor":
                assert isinstance(warmup_factor, float)
                self._set_lrs([base_lr * warmup_factor for base_lr in self.base_lrs])
            elif warmup_mode == "auto":
                assert isinstance(warmup_factor, float)
                self.warmup_end_lrs = self.regular_lrs_per_t[-1]
                self._set_lrs([base_lr * warmup_factor for base_lr in self.base_lrs])
            else:
                raise ValueError(f"Invalid warmup mode: {warmup_mode}")

    @property
    def _is_plateau(self):
        return isinstance(self.torch_scheduler, ReduceLROnPlateau)

    def _pre_compute_regular_lrs_per_t(self, max_t):
        regular_lrs_per_t = [self.base_lrs]
        if self._is_plateau:
            return regular_lrs_per_t * (max_t + 1)
        for _ in range(max_t):
            self.torch_scheduler.step()
            regular_lrs_per_t.append([param_group["lr"] for param_group in self.param_groups])
        return regular_lrs_per_t

    def _get_warmup_lrs(self, t, regular_lrs):
        # 为了简单，不再计算斜率，而是通过线性插值的方式获得warmup lr
        alpha = t / self.warmup_t
        if self.warmup_mode == "fix":
            return [self.warmup_init_lr * (1 - alpha) + base_lr * alpha for base_lr in self.base_lrs]
        elif self.warmup_mode == "factor":
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * factor for lr in regular_lrs]
        else:
            return [
                base_lr * self.warmup_factor * (1 - alpha) + end_lr * alpha
                for base_lr, end_lr in zip(self.base_lrs, self.warmup_end_lrs)
            ]

    def _set_lrs(self, lrs):
        if not isinstance(lrs, (list, tuple)):
            lrs = [lrs] * len(self.param_groups)
        for param_group, lr in zip(self.param_groups, lrs):
            param_group['lr'] = lr

    def epoch_update(self, metric=None):
        if not self.by_epoch:
            return

        self.last_epoch += 1
        if self.warmup_by_epoch and self.last_epoch < self.warmup_t:
            # 0 <= t < warmup_t时，根据不同的策略设置相应的warmup学习率
            self._set_lrs(self._get_warmup_lrs(self.last_epoch, self.regular_lrs_per_t[self.last_epoch]))
        elif self.warmup_by_epoch and self.last_epoch == self.warmup_t:
            # t == warmup_t时，将学习率恢复为常规学习率
            self._set_lrs(self.regular_lrs_per_t[-1])
        # in_iter_warmup=True时代表正在进行by iter的warmup，
        # lr已经被设置好了，此时torch_scheduler不能再执行step()函数修改lr
        elif not self.in_iter_warmup:
            if self._is_plateau:
                self.torch_scheduler.step(metric)
            else:
                self.torch_scheduler.step()

    def iter_update(self):
        if self.warmup_by_epoch:
            return

        self.last_iter += 1
        if self.last_iter < self.warmup_t:
            self.in_iter_warmup = True
            t = self.last_iter // self.epoch_len if self.by_epoch else self.last_iter
            self._set_lrs(self._get_warmup_lrs(self.last_iter, self.regular_lrs_per_t[t]))
        elif self.last_iter == self.warmup_t:
            self._set_lrs(self.regular_lrs_per_t[-1])
        else:
            # warmup结束之后，将in_iter_warmup变量置为False，此时torch_scheduler才可以进行正常的step()
            self.in_iter_warmup = False
            if not self.by_epoch:
                self.torch_scheduler.step()

    def state_dict(self):
        state = {key: value for key, value in self.__dict__.items() if key != "torch_scheduler"}
        state["torch_scheduler"] = self.torch_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.torch_scheduler.load_state_dict(state_dict.pop("torch_scheduler"))
        self.__dict__.update(state_dict)
