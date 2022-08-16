from __future__ import division


import sys
sys.path.append('..')
import os
import random
import argparse
from pathlib import Path
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm

from models import get_model
from datasets import get_dataset
from loss import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from models.superpoint import SuperPoint


if __name__ == '__main__':

    path = './checkpoints/7S-redkitchen-net1-initlr0.0005-iters30000-bsize1-aug1-light'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('net1', '7S', 64)
    model.init_weights()
    model.to(device)
    extractor = SuperPoint()
    extractor.to(device)
    extractor_path = os.path.join(path, 'extractor.pkl')
    mapping_path = os.path.join(path, 'model.pkl')
    extractor_state= torch.load(extractor_path, map_location=device)
    mapping = torch.load(mapping_path, map_location=device)
    extractor.load_state_dict(extractor_state['model_state'])
    model.load_state_dict(mapping['model_state'])

    torch.save(extractor.state_dict(), '{}/extractor.debug.pth'.format(path))
    torch.save(model.state_dict(), '{}/model.debug.pth'.format(path))

