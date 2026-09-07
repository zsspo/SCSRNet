from models.MSDCNN import*
from models.SSRNET import *
from models.TFNet import *
from models.TSFN import *
from models.DHIF import *
from models.fuseformer import *
from models.HSRNet import *
from models.PSRT import PSRTnet
from models.cyformer import Cyformer
from models.NGST import NGST_Net
from models.MCTNet import MCT

MODELS = {"MSD": MSDCNN,
          "DHIF": HSI_Fusion,
          "SSR": SSRNET,
          "TF": ResTFNet,
          "TSF": Net,
          "HSR": HSRNet,
          "fuseformer": Fuseformer,
          "PSRT": PSRTnet,
          "MCT": MCT,
          "NGST": NGST_Net,
          "Cyformer": Cyformer,
    }