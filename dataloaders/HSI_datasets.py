import collections
import math
import os
import random
from socket import MsgFlag

import cv2
import numpy as np
import torch
from torch.utils import data
from osgeo import gdal
import scipy
import scipy.ndimage
import scipy.io
import torchvision.transforms as transforms

# Pavia dataset (Dataset can be downloaded from the following link)
# http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
class pavia_dataset(data.Dataset):
    def __init__(
        self, config, train_split, is_dhp=False, want_DHP_MS_HR=False, resample=False
    ):
        self.split  = train_split        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["pavia_dataset"]["data_dir"] #Path to Pavia Center dataset 

        self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        
        self.images = [line.rstrip("\n") for line in open(self.file_list)] #Read image name corresponds to train/val/test set
    
        self.augmentation = self.config["pavia_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["pavia_dataset"]["LR_size"], self.config["pavia_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["pavia_dataset"]["HR_size"], self.config["pavia_dataset"]["HR_size"]]  #Size of the HR-HSI

        self.resam = resample
        cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        #PAN_image   = mat["pan"]
        PAN_image   = mat["msi"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["pavia_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")

            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]

            # Normalization
            #   MS_image    = torch.from_numpy(((np.array(MS_image) - self.dhp_mean)/self.dhp_std).transpose(2, 0, 1))
            #   PAN_image   = torch.from_numpy((np.array(PAN_image) - self.pan_mean)/self.pan_std)
            #   reference   = torch.from_numpy(((np.array(reference) - self.ref_mean)/self.ref_std).transpose(2, 0, 1))
        else:
            MS_image = mat["y"]

        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))  # CHW
        PAN_image   = torch.from_numpy((np.array(PAN_image)/1.0).transpose(2, 0, 1)) 
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))

        # Max Normalization
        MS_image    = MS_image/self.config["pavia_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["pavia_dataset"]["max_value"]
        reference   = reference/self.config["pavia_dataset"]["max_value"]
 
        if self.split == "train" and self.augmentation:
            MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    def __getitem__(self, index):
        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)
        return image_dict, MS_image, PAN_image, reference

# Botswana dataset
# http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
class botswana_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False, resample=False
    ):
        # Determine between train and val splits
        self.split = "train" if is_train else "val"

        #COnfig file
        self.config = config

        #Settings (DIP or no-DIP)
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp = is_dhp

        # Path to botswana dataset
        self.dir = self.config["botswana_dataset"]["data_dir"]

        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")
        self.resam = resample
        # List of all images
        self.images = [line.rstrip("\n") for line in open(self.file_list)]

        # augmentations (Not applicable in this experiment actually)
        self.augmentation = self.config["botswana_dataset"]["augmentation"]

        # Reading the LR crop size
        self.LR_crop_size = (self.config["botswana_dataset"]["LR_size"], self.config["botswana_dataset"]["LR_size"])

        # High resolution crop size
        self.HR_crop_size = [
            self.config["botswana_dataset"]["HR_size"],
            self.config["botswana_dataset"]["HR_size"],
        ]

        # To avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        # Set of all image names
        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
        # Reading the mat file
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]
        MS_image = mat["y"]
            
        # Convert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy((np.array(PAN_image)/1.0).transpose(2, 0, 1))
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["botswana_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["botswana_dataset"]["max_value"]
        reference   = reference/self.config["botswana_dataset"]["max_value"] 

        if self.resam:
            reference, MS_image, PAN_image = get_resample_HSI(index, reference, MS_image, PAN_image, self.files, self.images, self.split, self.dir)
        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)
        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    def __getitem__(self, index):
        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)
        return image_dict, MS_image, PAN_image, reference

# Chikusei dataset
# https://naotoyokoya.com/Download.html
# To directly using the provided dataloader, you can process the Pavia and Chikusei dataset through toolbox Hyperspec_Chikusei_MATLAB
# Naoto YOKOYA, 2016, http://park.itc.u-tokyo.ac.jp/sal/hyperdata/Hyperspec_Chikusei_MATLAB.zip
class chikusei_dataset(data.Dataset):
    def __init__(
        self, config, train_split, is_dhp=False, want_DHP_MS_HR=False, resample=False
    ):
        # Settings
        self.split          = train_split
        self.config         = config
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp         = is_dhp
        self.resam = resample
        # Paths
        self.dir            = self.config["chikusei_dataset"]["data_dir"]

        # Read train/val image indexes from the text file (train.txt, val.txt, train_dhp.txt)
      
        self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
           
        # list of all images
        self.images = [line.rstrip("\n") for line in open(self.file_list)]

        # augmentations
        self.augmentation = self.config["chikusei_dataset"]["augmentation"]

        self.LR_crop_size = (self.config["chikusei_dataset"]["LR_size"], self.config["chikusei_dataset"]["LR_size"])

        self.HR_crop_size = [
            self.config["chikusei_dataset"]["HR_size"],
            self.config["chikusei_dataset"]["HR_size"],
        ]

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        #PAN_image   = mat["pan"]
        PAN_image   = mat["msi"]

        if self.want_DHP_MS_HR:
            opt_lambda  = self.config["chikusei_dataset"]["optimal_lambda"]
            mat_dhp     = scipy.io.loadmat(image_dict["imgs"][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*opt_lambda))+ ".mat")
            
            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]
        else:
            MS_image = mat["y"]
            
        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy((np.array(PAN_image)/1.0).transpose(2, 0, 1))
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["chikusei_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["chikusei_dataset"]["max_value"]
        reference   = reference/self.config["chikusei_dataset"]["max_value"]           

        if self.resam:
            reference, MS_image, PAN_image = get_resample_HSI(index, reference, MS_image, PAN_image, self.files, self.images, self.split, self.dir)

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference


###  Botswana (x4) Dataset  ###
class botswana4_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False, want_DHP_MS_HR=False
    ):
        self.split  = "train" if is_train else "val"        #Define train and validation splits
        self.config = config                                #Configuration file
        self.want_DHP_MS_HR = want_DHP_MS_HR                #This ask: DO we need DIP up-sampled output as dataloader output?
        self.is_dhp         = is_dhp                        #This checks: "Is this DIP training?"
        self.dir = self.config["botswana4_dataset"]["data_dir"] #Path to Pavia Center dataset 
        
        if self.split == "val":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
        elif self.split == "train":
            self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")
            if is_dhp:
                self.file_list = os.path.join(self.dir, f"{self.split}_dhp" + ".txt")
        
        self.images = [line.rstrip("\n") for line in open(self.file_list)] #Read image name corresponds to train/val/test set
    
        self.augmentation = self.config["botswana4_dataset"]["augmentation"]   #Augmentation needed or not? 

        self.LR_crop_size = (self.config["botswana4_dataset"]["LR_size"], self.config["botswana4_dataset"]["LR_size"])  #Size of the the LR-HSI

        self.HR_crop_size = [self.config["botswana4_dataset"]["HR_size"], self.config["botswana4_dataset"]["HR_size"]]  #Size of the HR-HSI

        cv2.setNumThreads(0)    # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.img_root = self.dir+f+"/"
            self.files[self.split].append(
                {
                    "imgs": self.img_root + f + ".mat",
                }
            )

    def __len__(self):
        return len(self.files[self.split])
    
    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx==0:
            #Horizontal Flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
        elif aug_idx==1:
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])
        elif aug_idx==2:
            #Horizontal flip
            MS_image    = torch.flip(MS_image, [1]) 
            PAN_image   = torch.flip(PAN_image, [0])
            reference   = torch.flip(reference, [1])
            #Vertical Flip
            MS_image    = torch.flip(MS_image, [2])
            PAN_image   = torch.flip(PAN_image, [1])
            reference   = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]
       
        # read each image in list
        mat         = scipy.io.loadmat(image_dict["imgs"])
        reference   = mat["ref"]
        PAN_image   = mat["pan"]


        MS_image = mat["y"]
            
        # COnvert inputs into torch tensors
        MS_image    = torch.from_numpy((np.array(MS_image)/1.0).transpose(2, 0, 1))
        PAN_image   = torch.from_numpy(np.array(PAN_image)/1.0)
        reference   = torch.from_numpy((np.array(reference)/1.0).transpose(2, 0, 1))
        
        # Max Normalization
        MS_image    = MS_image/self.config["botswana4_dataset"]["max_value"]
        PAN_image   = PAN_image/self.config["botswana4_dataset"]["max_value"]
        reference   = reference/self.config["botswana4_dataset"]["max_value"]           

        #If split = "train" and augment = "true" do augmentation
        if self.split == "train" and self.augmentation:
            MS_image, PAN_image, reference = self._augmentaion(MS_image, PAN_image, reference)

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    
    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference


class dx_dataset(data.Dataset):
    def __init__(self, config, train_split, is_dhp=False, want_DHP_MS_HR=False):
        # Determine between train and val splits
        self.split = train_split

        # COnfig file
        self.config = config

        # Settings (DIP or no-DIP)
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp = is_dhp

        # Path to beijing_train dataset
        self.dir = self.config["dx_dataset"]["data_dir"]
        self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")

        # List of all images
        self.images = [line.strip("\n") for line in open(self.file_list)]

        # augmentations (Not applicable in this experiment actually)
        self.augmentation = self.config["dx_dataset"]["augmentation"]

        # Reading the LR crop size
        self.LR_crop_size = (self.config["dx_dataset"]["LR_size"], self.config["dx_dataset"]["LR_size"])

        # High resolution crop size
        self.HR_crop_size = [self.config["dx_dataset"]["HR_size"], self.config["dx_dataset"]["HR_size"], ]

        # To avoid Deadloack between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((159.0210, 175.6396, 165.8471), (56.6374, 56.5900, 60.2051)),
        ])


        # Set of all image names
        self.files = collections.defaultdict(list)
        for f in self.images:  #f为影像id
            self.files[self.split].append({"hrhsi": self.dir + 'ref/' + f + '.tif', "lrhsi": self.dir + 'hsi/' + f + '.tif', "msi": self.dir + 'rgb/' + f + '.tif'})

    def __len__(self):
        return len(self.files[self.split])

    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        if aug_idx == 0:
            # Horizontal Flip
            MS_image = torch.flip(MS_image, [1])
            PAN_image = torch.flip(PAN_image, [0])
            reference = torch.flip(reference, [1])
        elif aug_idx == 1:
            # Vertical Flip
            MS_image = torch.flip(MS_image, [2])
            PAN_image = torch.flip(PAN_image, [1])
            reference = torch.flip(reference, [2])
        elif aug_idx == 2:
            # Horizontal flip
            MS_image = torch.flip(MS_image, [1])
            PAN_image = torch.flip(PAN_image, [0])
            reference = torch.flip(reference, [1])
            # Vertical Flip
            MS_image = torch.flip(MS_image, [2])
            PAN_image = torch.flip(PAN_image, [1])
            reference = torch.flip(reference, [2])

        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]  #影像路径字典

        # Reading the tif file
        reference = gdal.Open(image_dict['hrhsi'])
        PAN_image = gdal.Open(image_dict['msi'])

        #mat = scipy.io.loadmat(image_dict["imgs"])
        #reference = mat["ref"]
        #PAN_image = mat["pan"]

        if self.want_DHP_MS_HR:
            opt_lambda = self.config["dx_dataset"]["optimal_lambda"]
            mat_dhp = scipy.io.loadmat(image_dict["imgs"][:-4] + "_dhp_" + "{0:0=1d}".format(int(10 * opt_lambda)) + ".mat")

            # Taking DIP up-sampled image as inputs
            MS_image = mat_dhp["dhp"]

            # Normalization  #   MS_image    = torch.from_numpy(((np.array(MS_image) - self.dhp_mean)/self.dhp_std).transpose(2, 0, 1))  #   PAN_image   = torch.from_numpy((np.array(PAN_image) - self.pan_mean)/self.pan_std)  #   reference   = torch.from_numpy(((np.array(reference) - self.ref_mean)/self.ref_std).transpose(2, 0, 1))
        else:
            MS_image = gdal.Open(image_dict['lrhsi'])

        im_width = reference.RasterXSize  # 栅格矩阵的列数
        im_height = reference.RasterYSize  # 栅格矩阵的行数
        im_bands = reference.RasterCount  # 波段数

        lim_width = MS_image.RasterXSize  # 栅格矩阵的列数
        lim_height = MS_image.RasterYSize  # 栅格矩阵的行数
        lim_bands = MS_image.RasterCount

        reference = reference.ReadAsArray(0, 0, im_width, im_height)  # 影像数组c, h, w
        MS_image = MS_image.ReadAsArray(0, 0, lim_width, lim_height)
        PAN_image = PAN_image.ReadAsArray(0, 0, im_width, im_height)

        # Convert inputs into torch tensors
        MS_image = torch.from_numpy((np.array(MS_image) / 1.0))
        PAN_image = torch.from_numpy(np.array(PAN_image) / 1.0)
        reference = torch.from_numpy(np.array(reference) / 1.0)

        #Max Normalization 归一化
        MS_image = MS_image / self.config["dx_dataset"]["max_value"]
        PAN_image = PAN_image / 255.0  #RGB
        #PAN_image = self.transform(PAN_image.transpose(1, 2, 0))
        reference = reference / self.config["dx_dataset"]["max_value"]

        # If split = "train" and augment = "true" do augmentation
        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference


class mdasn_dataset(data.Dataset):
    def __init__(self, config, train_split, is_dhp=False, want_DHP_MS_HR=False):
        # Determine between train and val splits
        self.split = train_split 

        # COnfig file
        self.config = config

        # Settings (DIP or no-DIP)
        self.want_DHP_MS_HR = want_DHP_MS_HR
        self.is_dhp = is_dhp

        # Path to beijing_train dataset
        self.dir = self.config["mdasn_dataset"]["data_dir"]

        self.file_list = os.path.join(self.dir, f"{self.split}" + ".txt")

        # List of all images
        self.images = [line.strip("\n") for line in open(self.file_list)]

        # augmentations (Not applicable in this experiment actually)
        self.augmentation = self.config["mdasn_dataset"]["augmentation"]

        # Reading the LR crop size
        self.LR_crop_size = (self.config["mdasn_dataset"]["LR_size"], self.config["mdasn_dataset"]["LR_size"])

        # High resolution crop size
        self.HR_crop_size = (self.config["mdasn_dataset"]["HR_size"], self.config["mdasn_dataset"]["HR_size"])

        # To avoid Deadloack between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

    
        # Set of all image names
        self.files = collections.defaultdict(list)
        #self.data_dir = self.dir + self.split + '\\'  #G:\hsi_data\Daxing_area_256x4\test\
        #self.data_dir = self.dir +  '\\'  #G:\hsi_data\Daxing_area_256x4\test\
        for f in self.images:  #f为影像id
            #self.files[self.split].append({"hrhsi": self.data_dir + 'ref\\' + f , "lrhsi": self.data_dir + 'hsi\\' + f , "msi": self.data_dir + 'rgb\\' + f })
            self.files[self.split].append({"hrhsi": self.dir + 'hrhsi/' + f , "lrhsi": self.dir + 'lrhsi/' + f , "hrmsi": self.dir + 'hrmsi/' + f })
            
    def __len__(self):
        return len(self.files[self.split])

    def _augmentaion(self, MS_image, PAN_image, reference):
        N_augs = 4
        aug_idx = torch.randint(0, N_augs, (1,))
        return MS_image, PAN_image, reference

    def getHSIdata(self, index):
        image_dict = self.files[self.split][index]  #影像路径字典

        # Reading the tif file
        reference = gdal.Open(image_dict['hrhsi'])
        PAN_image = gdal.Open(image_dict['hrmsi'])

        #mat = scipy.io.loadmat(image_dict["imgs"])
        #reference = mat["ref"]
        #PAN_image = mat["pan"]

    
        MS_image = gdal.Open(image_dict['lrhsi'])

        im_width = reference.RasterXSize  # 栅格矩阵的列数
        im_height = reference.RasterYSize  # 栅格矩阵的行数
        im_bands = reference.RasterCount  # 波段数

        lim_width = MS_image.RasterXSize  # 栅格矩阵的列数
        lim_height = MS_image.RasterYSize  # 栅格矩阵的行数
        lim_bands = MS_image.RasterCount

        reference = reference.ReadAsArray(0, 0, im_width, im_height)  # 影像数组c, h, w
        MS_image = MS_image.ReadAsArray(0, 0, lim_width, lim_height)
        PAN_image = PAN_image.ReadAsArray(0, 0, im_width, im_height)

        # Convert inputs into torch tensors
        MS_image = torch.from_numpy(np.array(MS_image) )
        PAN_image = torch.from_numpy(np.array(PAN_image))
        reference = torch.from_numpy(np.array(reference))

        #Max Normalization 归一化
        MS_image = MS_image / self.config["mdasn_dataset"]["max_value"]
        PAN_image = PAN_image / self.config["mdasn_dataset"]["max_value"]
        reference = reference / self.config["mdasn_dataset"]["max_value"]
   

        # If split = "train" and augment = "true" do augmentation
        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        return image_dict, MS_image, PAN_image, reference

    def __getitem__(self, index):

        image_dict, MS_image, PAN_image, reference = self.getHSIdata(index)

        return image_dict, MS_image, PAN_image, reference



