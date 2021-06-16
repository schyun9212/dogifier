from yacs.config import CfgNode as CN

_C = CN()

################################################
# Model Configs
_C.MODEL = CN()
_C.MODEL.TYPE = "dino" # ("timm", "dino")
_C.MODEL.NUM_LABELS = 1000
_C.MODEL.LR = 0.001

# timm config
_C.MODEL.TIMM = CN()
_C.MODEL.TIMM.NAME = "vit_base_patch16_224" # Vision Transformers in timm.list_models()[398:425]

# dino config
_C.MODEL.DINO = CN()
_C.MODEL.DINO.NAME = "dino_vits16"

def get_cfg():
    return _C.clone()
