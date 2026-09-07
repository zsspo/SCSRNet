
import os
import sys
import argparse
import shutil
import json
import torch
import time
import datetime
import copy
import numpy as np
from torch.nn.functional import threshold, unfold
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Iterable, Optional

from dataloaders.HSI_datasets import *
from utils.logger import Logger
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
from utils.metrics import *
from utils.vgg_perceptual_loss import VGGPerceptualLoss, VGG19
from utils.spatial_loss import Spatial_Loss
from utils.tensor_rotate import rotate_tensor
from utils.optim_factory import create_optimizer, LayerDecayValueAssigner
from utils import utils

from models.MSDCNN import*

from timm.utils import ModelEma
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import savemat

from models.model import MODELS
from models.SCSRNet import SCSRNet

import wandb
import wandb
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Set proxy environment variables
#os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
#os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# Set W&B to offline mode
os.environ["WANDB_MODE"] = "offline"

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
        
def get_rgb_normalized(r, g, b):
    max_r = np.max(np.max(r, axis=1), axis=0)
    max_g = np.max(np.max(g, axis=1), axis=0)
    max_b = np.max(np.max(b, axis=1), axis=0)
    r = r/max_r*255.0
    g = g/max_g*255.0
    b = b/max_b*255.0
    img_rgb = np.stack((b, g, r), axis=2)
    img_rgb[img_rgb<0.0]=0.0
    return img_rgb

def get_mean_std():
    transform = transforms.Compose(
        [
      
            transforms.ToTensor(),
        ]
    )
    data_loader = torch.utils.data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, is_train=True, want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=config["train_batch_size"],
            num_workers=config["num_workers"], shuffle=True,
            pin_memory=False,drop_last=True)

    nb_samples = 0.
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    for i, data in enumerate(data_loader, 0):
        # Reading data
        _, MS_image, images, reference = data
        N, C, H, W = images.shape[:4]
        data = images.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N

    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)


def data_load_test_show(MS_image, PAN_image, reference):
    r_img = MS_image[0, 97, :, :].squeeze(0).numpy()
    g_img = MS_image[0, 57, :, :].squeeze(0).numpy()
    b_img = MS_image[0, 23, :, :].squeeze(0).numpy()

    rgb_hsi = np.uint8(get_rgb_normalized(r_img, g_img, b_img))

    r_img = PAN_image[0, 2, :, :].squeeze(0).numpy()
    g_img = PAN_image[0, 1, :, :].squeeze(0).numpy()
    b_img = PAN_image[0, 0, :, :].squeeze(0).numpy()

    rgb_msi = np.uint8(get_rgb_normalized(r_img, g_img, b_img))

    r_ref = reference[0, 97, :, :].squeeze(0).numpy()
    g_ref = reference[0, 57, :, :].squeeze(0).numpy()
    b_ref = reference[0, 23, :, :].squeeze(0).numpy()

    rgb_ref = np.uint8(get_rgb_normalized(r_ref, g_ref, b_ref))

    plt.subplot(3, 1, 1)
    plt.imshow(rgb_msi)
    plt.title('msi')
    plt.xticks(), plt.yticks()

    plt.subplot(3, 1, 2)
    plt.imshow(rgb_hsi)
    plt.title('hsi')
    plt.xticks(), plt.yticks()

    plt.subplot(3, 1, 3)
    plt.imshow(rgb_ref)
    plt.title('ref')
    plt.xticks(), plt.yticks()

    plt.show()
    plt.close()


            
# TRAIN EPOCH
def train(model: torch.nn.Module, criterion: torch.nn.Module,
                    train_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
   
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5
    
    optimizer.zero_grad()
    
    for data_iter_step, data in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Reading data
        _, MS_image, PAN_image, reference = data
        
        test = False
        # 测试图像宽高是否
        if test:
            data_load_test_show(MS_image, PAN_image, reference)

        # Taking model outputs ...
        #if epoch==1: 
             #print(MS_image.max())
             #print(MS_image.min())
        MS_image    = Variable(MS_image.float().cuda()) 
        PAN_image   = Variable(PAN_image.float().cuda()) 
        out         = model(MS_image, PAN_image)
        outputs = out["pred"]

        # Normal L1 loss
        if config[config["train_dataset"]]["Normalized_L1"]:
            max_ref     = torch.amax(reference, dim=(2,3)).unsqueeze(2).unsqueeze(3).expand_as(reference).cuda()
            loss        = criterion(outputs/max_ref, to_variable(reference)/max_ref)
        else:
            loss        = criterion(outputs, to_variable(reference))

        # Spatial loss
        if config[config["train_dataset"]]["Spatial_Loss"]:
            loss += config[config["train_dataset"]]["Spatial_Loss_F"]*Spatial_loss(to_variable(reference), outputs)
        
        loss_value = loss.item()
        loss = loss/update_freq
        torch.autograd.backward(loss)
        if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        
        if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.set_step()

    writer.add_scalar('Loss/train', loss, epoch)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
 
# TEST EPPOCH
@torch.no_grad()
def test(data_loader, model, criterion):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    loss   = 0.0
    cc          = 0.0
    sam         = 0.0
    SAM_value   = 0.0
    rmse        = 0.0
    RMSE_value  = 0.0
    ergas       = 0.0
    ERGAS_value = 0.0
    psnr        = 0.0
    PSNR_value  = 0.0
    pred_dic = {}
    
    # switch to evaluation mode
    model.eval()
    for i, data in enumerate(metric_logger.log_every(data_loader, 10, header)):
        image_dict, MS_image, PAN_image, reference = data
        # Inputs and references...
        MS_image    = MS_image.float().cuda().contiguous()
        PAN_image   = PAN_image.float().cuda().contiguous()
        reference   = reference.float().cuda().contiguous()

        # Taking model output
        out     = model(MS_image, PAN_image)
        outputs = out["pred"]

        # Normal L1 loss
        if config[config["train_dataset"]]["Normalized_L1"]:
            max_ref     = torch.amax(reference, dim=(2,3)).unsqueeze(2).unsqueeze(3).expand_as(reference).cuda()
            loss        = criterion(outputs/max_ref, to_variable(reference)/max_ref)
        else:
            loss        = criterion(outputs, to_variable(reference))

        # Spatial loss
        if config[config["train_dataset"]]["Spatial_Loss"]:
            loss += config[config["train_dataset"]]["Spatial_Loss_F"]*Spatial_loss(to_variable(reference), outputs)
        
        # Scalling
        outputs[outputs<0]      = 0.0
        outputs[outputs>1.0]    = 1.0
        outputs                 = torch.round(outputs*config[config["train_dataset"]]["max_value"])

        # match the key of different datasets
        if  config['train_dataset'] =="dx_dataset" or config['train_dataset'] =="mdasn_dataset":
            pred_dic.update({image_dict["hrhsi"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
        elif config['train_dataset'] == "mdas_dataset":
            pred_dic.update({image_dict[0]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
        else:
            pred_dic.update({image_dict["imgs"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
            
        reference = torch.round(reference.detach()*config[config["train_dataset"]]["max_value"])
    
        ### Computing performance metrics ###
        # Cross-correlation
        CC_value = cross_correlation(outputs, reference)
        cc += CC_value
        # SAM
        SAM_value = SAM(outputs, reference)
        sam += SAM_value
        # RMSE
        RMSE_value = RMSE(outputs/torch.max(reference), reference/torch.max(reference))
        rmse += RMSE_value
        # ERGAS
        beta = torch.tensor(config[config["train_dataset"]]["HR_size"]/config[config["train_dataset"]]["LR_size"]).cuda()
        ERGAS_value = ERGAS(outputs, reference, beta)
        ergas += ERGAS_value
        # PSNR
        PSNR_value = PSNR(outputs, reference)
        psnr += PSNR_value
        
        batch_size = MS_image.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['PSNR'].update(PSNR_value.item(), n=batch_size)
        metric_logger.meters['SAM'].update(SAM_value.item(), n=batch_size)
        metric_logger.meters['RMSE'].update(RMSE_value.item(), n=batch_size)
        metric_logger.meters['ERGAS'].update(ERGAS_value.item(), n=batch_size)
        metric_logger.meters['CC'].update(CC_value.item(), n=batch_size)
        
    # Taking average of performance metrics over test set
    cc /= len(data_loader)
    sam /= len(data_loader)
    rmse /= len(data_loader)
    ergas /= len(data_loader)
    psnr /= len(data_loader)
    
    # Writing test results to tensorboard
    writer.add_scalar('Loss/Val', loss, epoch)
    writer.add_scalar('Val_Metrics/CC', cc, epoch)
    writer.add_scalar('Val_Metrics/SAM', sam, epoch)
    writer.add_scalar('Val_Metrics/RMSE', rmse, epoch)
    writer.add_scalar('Val_Metrics/ERGAS', ergas, epoch)
    writer.add_scalar('Val_Metrics/PSNR', psnr, epoch)

    #Normalizing the images
    outputs     = outputs/torch.max(reference)
    reference   = reference/torch.max(reference)
    MS_image    = MS_image/torch.max(reference)
 
    #Return Outputs
    metrics = { "loss": float(loss), 
                "cc": float(cc), 
                "sam": float(sam), 
                "rmse": float(rmse), 
                "ergas": float(ergas), 
                "psnr": float(psnr)}
    
    print('* PSNR {psnr.global_avg:.6f} CC {cc.global_avg:.6f} SAM {sam.global_avg:.6f} ERGAS {ergas.global_avg:.6f} RMSE {rmse.global_avg:.6f} Loss {loss.global_avg:.4f}'
          .format(psnr=metric_logger.PSNR, cc=metric_logger.CC, sam=metric_logger.SAM, ergas=metric_logger.ERGAS, rmse=metric_logger.RMSE, loss=metric_logger.loss))
    return image_dict, pred_dic, metrics, {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
      

    __dataset__ = {"pavia_dataset":    pavia_dataset, "botswana_dataset": botswana_dataset,
                   "chikusei_dataset": chikusei_dataset, "botswana4_dataset": botswana4_dataset, "road_train_dataset": road_train_dataset, "dx_dataset": dx_dataset,"dx1_dataset": dx1_dataset, "mdas_dataset": mdas_dataset,"mdasn_dataset": mdasn_dataset
        }

    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config_lk/config_rgb_lk_dx.json', type=str, help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--update_freq', default=2, type=int,
                        help='gradient accumulation steps')
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=True, help='Using ema to eval during training.')
    
    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    # LR parameters
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    
    args = parser.parse_args()

    # LOADING THE CONFIG FILE
    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True

    # Initialize a W&B run
    wandb.init(project=config["name"], name=config["experim_name"], settings=wandb.Settings(init_timeout=120))
    
    # SEEDS
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)

    # NUMBER OF GPUs
    num_gpus = torch.cuda.device_count()
    
    # Dataset loading
    print("Training with dataset => {}".format(config["train_dataset"]))

    train_loader = data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, "train", want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=config["train_batch_size"],
            num_workers=config["num_workers"], shuffle=True,
            pin_memory=False,drop_last=True)

    val_loader = data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, "val", want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=config["val_batch_size"],
            num_workers=config["num_workers"], shuffle=True, pin_memory=False, )
    
    # if not using testing set, set the split tag of test_loader to 'val'
    test_loader = data.DataLoader(
            __dataset__[config["train_dataset"]](
                    config, "val", want_DHP_MS_HR=config["is_DHP_MS"], ), batch_size=config["val_batch_size"],
            num_workers=config["num_workers"], shuffle=True, pin_memory=False, )
    
    # MODEL 
    #model = MODELS[config["model"]](config)
    model = SCSRNet(config=config)
    print(f'\n{model}\n')
   
    # SENDING MODEL TO DEVICE
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model).cuda()
    else:
        print(torch.cuda.is_available())
        print("Single Cuda Node is avaiable")
        model.cuda()
        
    #Model EMA
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    
    
    total_batch_size = config["train_batch_size"]*args.update_freq   #4*2
    num_training_steps_per_epoch = len(train_loader) // total_batch_size  # avoid len(train_loader) < total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(train_loader))
    print("Number of training per epoch = %d" % num_training_steps_per_epoch)

    # INITIALIZATION OF PARAMETERS
    start_epoch = 1
    total_epochs = config["trainer"]["total_epochs"]

    # OPTIMIZER： SGD, ADAM, COS
    # if config["optimizer"]["type"] == "SGD":
    #     optimizer = optim.SGD(
    #             model.parameters(), lr=config["optimizer"]["args"]["lr"],
    #             momentum=config["optimizer"]["args"]["momentum"],
    #             weight_decay=config["optimizer"]["args"]["weight_decay"]
    #             )
    # elif config["optimizer"]["type"] == "ADAM":
    #     optimizer = optim.Adam(
    #             model.parameters(), lr=config["optimizer"]["args"]["lr"],
    #             weight_decay=config["optimizer"]["args"]["weight_decay"]
    #             )
    # else:
    #     exit("Undefined optimizer type")
  
    assigner = None
    total_epochs = config["trainer"]["total_epochs"]
    # Optimizer
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
      
    # Lr_schedule
    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, total_epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, total_epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # SETTING UP TENSORBOARD and COPY JSON FILE TO SAVE DIRECTORY
    PATH = "./" + config["experim_name"] + "/" + config["model"] + '/' + config["train_dataset"]
    LOG_PATH = PATH + "/logs/"
    ensure_dir(PATH + "/")
    ensure_dir(LOG_PATH)
    writer = SummaryWriter(log_dir=PATH)
    log_writer = utils.TensorboardLogger(log_dir=LOG_PATH) #tensorboard
    shutil.copy2(args.config, PATH)

    # Print model to text file
    original_stdout = sys.stdout
    with open(PATH + "/" + "model_summary.txt", 'w+') as f:
        sys.stdout = f
        print(f'\n{model}\n')
        sys.stdout = original_stdout

    # LEARNING RATE SHEDULER
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["optimizer"]["step_size"], gamma=config["optimizer"]["gamma"])
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=config["optimizer"]["gamma"])
    # IF RESUME
    if args.resume is not None:
        print("Loading from existing FCN and copying weights to continue....")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=False)
    #else:
        #initialize_weights(model)
  
    # LOSS
    if config[config["train_dataset"]]["loss_type"] == "L1":
        criterion = torch.nn.L1Loss()
        HF_loss = torch.nn.L1Loss()
    elif config[config["train_dataset"]]["loss_type"] == "MSE":
        criterion = torch.nn.MSELoss()
        HF_loss = torch.nn.MSELoss()
    else:
        exit("Undefined loss data_loader_val")

    if config[config["train_dataset"]]["Spatial_Loss"]:
        Spatial_loss = Spatial_Loss(in_channels=config[config["train_dataset"]]["spectral_bands"]).cuda()

    # MAIN LOOP
    best_psnr = 0.0
    if args.model_ema and args.model_ema_eval:
        best_psnr_ema = 0.0
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            
        train_stats = train(
            model, criterion, train_loader, optimizer,
            epoch, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        )

        # Log training metrics for each epoch
        wandb.log({"epoch": epoch, "train_loss": train_stats['loss'], "train_lr": train_stats['lr']})

        # validation, select best model
        image_dict, pred_dic, metrics, val_stats = test(val_loader, model, criterion)  
         # Log testing metrics for each epoch
        wandb.log({"epoch": epoch, "val_PSNR": metrics['psnr'], "val_CC": metrics['cc'], "val_SAM": metrics['sam'], "val_RMSE": metrics['rmse'], "val_ERGAS": metrics['ergas']})
        print(f"PSNR of the model on the {len(val_loader)} val images: {metrics['psnr']:.6f} dB")
        if best_psnr < metrics['psnr']:
            best_model = copy.deepcopy(model)  
            best_psnr = metrics['psnr']
            utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    epoch="best", model_ema=model_ema, out_dir=PATH)
            torch.save(model.state_dict(), PATH + "/" + "best_val_model.pth")
            with open(PATH + "/" + "best_val_metrics.json", "w+") as outfile:
                json.dump(metrics, outfile)
        print(f'Best PSNR: {best_psnr:.6f} dB')

        if log_writer is not None:
            log_writer.update(val_psnr=val_stats['PSNR'], head="perf", step=epoch)
            log_writer.update(val_sam=val_stats['SAM'], head="perf", step=epoch)
            log_writer.update(val_rmse=val_stats['RMSE'], head="perf", step=epoch)
            log_writer.update(val_ergas=val_stats['ERGAS'], head="perf", step=epoch)
            log_writer.update(val_cc=val_stats['CC'], head="perf", step=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        # repeat testing routines for EMA, if ema eval is turned on
        if args.model_ema and args.model_ema_eval:
            image_dict, pred_dic_ema, metrics_ema, val_stats_ema = test(val_loader, model_ema.ema, criterion)
            print(f"PSNR of the model EMA on {len(val_loader)} val images: {metrics_ema['psnr']:.6f} dB")
            if best_psnr_ema < metrics_ema['psnr']:
                best_psnr_ema = metrics_ema['psnr']
                #utils.save_model(
                    #args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    #epoch="best-ema", model_ema=model_ema, out_dir=PATH)
                #torch.save(model_ema.state_dict(), PATH + "/" + "best_model_ema.pth")
                with open(PATH + "/" + "best_val_metrics_ema.json", "w+") as outfile:
                    json.dump(metrics_ema, outfile)
                print(f'Max EMA PSNR: {best_psnr_ema:.6f} dB')
                
            if log_writer is not None:
                log_writer.update(val_psnr_ema=val_stats_ema['PSNR'], head="perf", step=epoch)
                log_writer.update(val_sam_ema=val_stats_ema['SAM'], head="perf", step=epoch)
                log_writer.update(val_rmse_ema=val_stats_ema['RMSE'], head="perf", step=epoch)
                log_writer.update(val_ergas_ema=val_stats_ema['ERGAS'], head="perf", step=epoch)
                log_writer.update(val_cc_ema=val_stats_ema['CC'], head="perf", step=epoch)

            log_stats.update({**{f'_{k}_ema': v for k, v in val_stats_ema.items()}})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))  
    
    #testing, using the best model
    image_dict, pred_dic, test_metrics, test_stats = test(test_loader, best_model, criterion)  
    log_stats.update({**{f'_{k}_final_eval': v for k, v in test_stats.items()}})
    with open(PATH + "/" + "test_metrics.json", "w+") as outfile:
                json.dump(test_metrics, outfile)
    if log_writer is not None:
        log_writer.flush()
    with open(os.path.join(LOG_PATH, "train_val_test_log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")
        f.write('Training time {}'.format(total_time_str) + "\n")
    print(f"Test PSNR of the model on the {len(test_loader)} test images: {test_metrics['psnr']:.6f} dB")

    # Saving best prediction
    savemat(PATH + "/" + "final_prediction.mat", pred_dic)
