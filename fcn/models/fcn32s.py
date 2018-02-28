# 32x fcn
import os.path as osp

import fcn
import numpy as np
import torch
import torch.nn as nn


class FCN32s(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn32s_from_caffe.pth')

