import argparse
import os
import sys

from pyssr.core.io import cache_url
from yacs.config import CfgNode as CfgNode


# Global config object
_C = CfgNode()

# Example usage:
#   from core.config import cfg
cfg = _C


# ------------------------------------------------------------------------------------ #
# Model options
# ------------------------------------------------------------------------------------ #
_C.MODEL = CfgNode()

# Model type
_C.MODEL.TYPE = ""

# Number of input channels
_C.MODEL.IN_CHANNELS = 1

# Loss function (see pyssr/models/loss.py for options)
_C.MODEL.LOSS_FUN = "mse"
_C.MODEL.VIS_THRESHOLD = 5


# ------------------------------------------------------------------------------------ #
# SUBPIXEL options
# ------------------------------------------------------------------------------------ #
_C.SUBPIXEL = CfgNode()

_C.SUBPIXEL.UPSCALE_FACTOR = 2


# ------------------------------------------------------------------------------------ #
# U-Net options
# ------------------------------------------------------------------------------------ #
_C.UNET = CfgNode()

_C.UNET.NUM_CHANS = 32
_C.UNET.NUM_POOLS = 2
_C.UNET.DROP_PROB = 0.0


# ------------------------------------------------------------------------------------ #
# Batch norm options
# ------------------------------------------------------------------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = True
_C.BN.NUM_SAMPLES_PRECISE = 8192

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0


# ------------------------------------------------------------------------------------ #
# Optimizer options
# ------------------------------------------------------------------------------------ #
_C.OPTIM = CfgNode()

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Exponential decay factor
_C.OPTIM.GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0


# ------------------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------------------ #
_C.TRAIN = CfgNode()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Image size
_C.TRAIN.IM_SIZE = 224

# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Weights to start training from
_C.TRAIN.WEIGHTS = ""


# ------------------------------------------------------------------------------------ #
# Testing options
# ------------------------------------------------------------------------------------ #
_C.TEST = CfgNode()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200

# Image size
_C.TEST.IM_SIZE = 256

# Weights to use for testing
_C.TEST.WEIGHTS = ""


# ------------------------------------------------------------------------------------ #
# Common train/test data loader options
# ------------------------------------------------------------------------------------ #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ------------------------------------------------------------------------------------ #
# Memory options
# ------------------------------------------------------------------------------------ #
_C.MEM = CfgNode()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True


# ------------------------------------------------------------------------------------ #
# CUDNN options
# ------------------------------------------------------------------------------------ #
_C.CUDNN = CfgNode()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = True


# ------------------------------------------------------------------------------------ #
# Precise timing options
# ------------------------------------------------------------------------------------ #
_C.PREC_TIME = CfgNode()

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ------------------------------------------------------------------------------------ #
# Misc options
# ------------------------------------------------------------------------------------ #

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = "/tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port range for multi-process groups (actual port selected randomly)
_C.HOST = "localhost"
_C.PORT_RANGE = [10000, 65000]

# Models weights referred to by URL are downloaded to this local cache
_C.DOWNLOAD_CACHE = "/tmp/pyssr-download-cache"


# ------------------------------------------------------------------------------------ #
# Default config
# ------------------------------------------------------------------------------------ #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg(cache_urls=True):
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
    if cache_urls:
        cache_cfg_urls()


def cache_cfg_urls():
    """Download URLs in config, cache them, and rewrite cfg to use cached file."""
    _C.TRAIN.WEIGHTS = cache_url(_C.TRAIN.WEIGHTS, _C.DOWNLOAD_CACHE)
    _C.TEST.WEIGHTS = cache_url(_C.TEST.WEIGHTS, _C.DOWNLOAD_CACHE)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file", help=help_s, required=True, type=str)
    help_s = "See pyssr/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)
