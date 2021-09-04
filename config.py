import os
import time
from easydict import EasyDict as edict


__C = edict()
ActivityConfig = __C

# experiment setting
__C.TIMESTAMP = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())  # TODO: set time stamp
__C.MODEL_NAME = 'c3d'  # TODO: set training model
__C.MODEL_LIST = ['c3d', 'r3d', 'r21d']
__C.SAVE_PATH = '/data/guojie'
__C.GPUS = 4  # TODO: set the count of used GPU
__C.GPU_ID = '0,1,2,3'  # TODO: set available gpus id
__C.MULTI_GPU = True  # TODO: set use multi-gpus or not
__C.RANDOM_SEED = 632  # TODO: set random seed
__C.SHOW_INFO = 20  # TODO: set how many steps pass that show information
__C.CHECK_EPOCH = 20  # TODO: set how many epochs pass that save checkpoint
__C.APEX = False  # TODO: set use APEX to accelerate training or not


# train setting
__C.TRAIN = edict()
__C.TRAIN.TYPE = 'SSL'  # TODO: set training type: [SSL, Finetune]
__C.TRAIN.EPOCH = 300  # TODO: set training epoch
__C.TRAIN.START_EPOCH = 1  # TODO: set the strat epoch
__C.TRAIN.BATCH_SIZE = 8  # TODO: set batch size
__C.TRAIN.NUM_WORKERS = 4  # TODO: set num_workers
__C.TRAIN.LEARNING_RATE = 0.01 # TODO: set training learning rate
__C.TRAIN.MOMENTUM = 0.9  # TODO: set momentum parameter
__C.TRAIN.WEIGHT_DECAY = 0.0005 # TODO: set weight decay rate
__C.TRAIN.MIN_LR = 1e-7  # TODO: set minimum learning rate
__C.TRAIN.PATIENCE=50  # TODO: set patience rate
__C.TRAIN.FACTOR = 0.1  # TODO: set factor parameter
__C.TRAIN.FUSION = 'weight-average'  # TODO: set result fusion method
__C.TRAIN.RGB_WEIGHT = 0.5  # TODO: set weight of RGB stream
__C.TRAIN.DIFF_WEIGHT = 0.5  # TODO: set weight of diff stream


# SSL training setting
__C.TRAIN.FRAME_DIFF = edict()
__C.TRAIN.FRAME_DIFF.REQUIREMENT = True  # TODO: set requirements for rgb_diff
__C.TRAIN.FRAME_DIFF.LOW = 0.8  # TODO: set hyper-parameter in rgb_diff
__C.TRAIN.FRAME_DIFF.HIGH = 2.0  # TODO: set hyper-parameter in rgb_diff


# dataset setting
__C.DATASET = edict()
__C.DATASET.LOAD_TYPE = 'DataLoaderX'  # TODO: set type of data loader, e.g., normal or DataLoaderX
__C.DATASET.ROOT_PATH = "/home/guojie/Dataset"  # TODO: set dataset root path
__C.DATASET.NAME = "UCF-101-origin"  # TODO: set used dataset
__C.DATASET.CLASS_NUM = 8
__C.DATASET.VAL_SIZE = 800


__C.EXP_TAG = __C.TIMESTAMP + "_" + __C.MODEL_NAME + "_" + __C.TRAIN.TYPE