'''

    This creates the whole pre-processing pipeline.

'''

import torch
import torch.nn as nn
import numpy as np
from functools import partial

import onmt
import onmt.inputters
import onmt.modules
import onmt.utils
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.decoders.decoder import InputFeedRNNDecoder
from onmt.utils.misc import tile
import onmt.translate

from tqdm import tqdm
from typing import Callable
from utils.goodies import *
import time
import utils.tensor_utils as tu
from onmt.inputters.inputter import build_dataset_iter, \
    load_fields, _collect_report_features


