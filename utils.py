from __future__ import division

import os
import torch
import numpy as np
import random

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def adjust_lr(optimizer, init_lr, c_iter, n_iter):
    lr = init_lr * (0.5 ** ((c_iter + 200000 - n_iter) // 50000 + 1 if (c_iter 
         + 200000 - n_iter) >= 0 else 0))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr    

def save_state(savepath, epoch, classifier, extractor, optimizer, suffix=None):
    state = {'epoch': epoch,
        'model_state': classifier.state_dict(),
        'optimizer_state': optimizer.state_dict()}
    state_extr =  {'epoch': epoch,
        'model_state': extractor.state_dict(),
        'optimizer_state': optimizer.state_dict()}
    if suffix is None:
        filepath = os.path.join(savepath, 'classifier.pkl')
        filepath_extract = os.path.join(savepath, 'extractor.pkl')
    else:
        filepath = os.path.join(savepath, 'classifier_{}.pkl'.format(suffix))
        filepath_extract = os.path.join(savepath, 'extractor_{}.pkl'.format(suffix))
    torch.save(state, filepath)
    torch.save(state_extr, filepath_extract)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     #torch.backends.cudnn.deterministic = True
     
