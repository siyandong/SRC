from __future__ import division

#import os
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


def get_optimizer(model, state=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if state is not None: optimizer.load_state_dict(state)
    return optimizer

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_loss(prediction, labels, mask):
    cls_loss = CELoss()
    return cls_loss(prediction, labels, mask)

def do_learning(extractor, model, optimizer, loader, k_step, device):
    model.train()
    acc_list = []
    loss_list = []
    for idx, (img, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh) in enumerate(loader):
        if idx >= k_step: break
        if mask.sum() == 0: continue
        # forward.
        img = img.to(device)
        mask = mask.to(device)
        lbl_1 = lbl_1.to(device)
        lbl_2 = lbl_2.to(device)
        lbl_1_oh = lbl_1_oh.to(device)
        lbl_2_oh = lbl_2_oh.to(device)
        feature_map = extractor(img)
        lbl_2_pred, lbl_1_pred = model(feature_map, lbl_1_oh, lbl_2_oh)
        # loss.
        lbl_1_loss = get_loss(lbl_1_pred, lbl_1, mask)
        lbl_2_loss = get_loss(lbl_2_pred, lbl_2, mask)
        loss = lbl_1_loss + lbl_2_loss
        # backward and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record accuracy.
        lbl_1_p = torch.argmax(lbl_1_pred, dim=1)  
        lbl_2_p = torch.argmax(lbl_2_pred, dim=1)  
        lbl_p = (lbl_1_p * args.n_class + lbl_2_p)  
        lbl_gt = (lbl_1 * args.n_class + lbl_2)
        idx = torch.eq(lbl_p, lbl_gt)
        mask_new = mask.mul(idx)
        accuracy = torch.sum(mask_new)/ torch.sum(mask) 
        acc_list.append(accuracy.item())
        loss_list.append(loss.item())
    return acc_list, loss_list


def train_reptile(extractor, meta_optimizer, meta_model, state, loader, k_step, device):
    # clone model.
    model = meta_model.clone()
    optimizer = get_optimizer(model, state)
    # update the fast nets.
    acc_list, loss_list = do_learning(extractor, model, optimizer, loader, k_step, device)
    #print('acc {:.2f}%'.format(acc_list[-1]*100))
    state = optimizer.state_dict()
    # update slow net.
    meta_model.point_grad_to(model)
    meta_optimizer.step()
    return acc_list, loss_list

def train_common(extractor, meta_optimizer, meta_model, state, loader, k_step, device):
    acc_list, _ = do_learning(extractor, meta_model, meta_optimizer, loader, k_step, device)
    print('acc {:.2f}%'.format(acc_list[-1]*100))


def train(args):
    
    # prepare datasets. 
    dataset = get_dataset('12S')
    dataset_train = dict()
    loader_train = dict()
    for scene in args.scenes:
        dataset_train[scene] = dataset(n_class=args.n_class, root=args.data_path, info=args.training_info, scene=scene, aug=args.aug)
        loader_train[scene] = data.DataLoader(dataset_train[scene], batch_size=args.batch_size, num_workers=4, shuffle=True)

    cls_loss = CELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = SuperPoint()
    extractor.to(device)
    meta_model = get_model(args.model, args.n_class)
    meta_model.init_weights()
    meta_model.to(device)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr, eps=1e-8, betas=(0.9, 0.999))


    # checkpoint.
    model_id = "pretrain_{}_{}".format('12S', args.model, args.train_id)
    save_path = Path(model_id)
    args.save_path = 'checkpoints'/save_path
    args.save_path.mkdir(parents=True, exist_ok=True)
    

    # start training.
    train_scenes = []
    for scene in args.scenes:
        train_scenes.append(scene)
    state = None
    for outer_iter_idx in tqdm(range(args.n_iter)):

        # update learning rate.
        meta_lr = args.meta_lr * (1. - outer_iter_idx/float(args.n_iter))
        set_learning_rate(meta_optimizer, meta_lr)

        # train.
        scene = random.choice(train_scenes)

        acc_list, loss_list = train_reptile(
            extractor=extractor, 
            meta_optimizer=meta_optimizer, 
            meta_model=meta_model, 
            state=state, 
            loader=loader_train[scene], 
            k_step=args.k_step,
            device=device)

        if outer_iter_idx % 20 == 0:
            if len(acc_list)>1:
                writer.add_scalar("Acc/train", np.mean(acc_list), outer_iter_idx)
            if len(loss_list)>1:
                writer.add_scalar("Loss/train", np.mean(loss_list), outer_iter_idx)

        # save checkpoint.
        if outer_iter_idx % 2000 == 0:
            save_state(args.save_path, outer_iter_idx, meta_model, extractor, meta_optimizer, suffix=outer_iter_idx)


    #writer.flush()
    save_state(args.save_path, outer_iter_idx, meta_model, extractor, meta_optimizer,suffix=None)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre-train the SRC model on 12-Scenes dataset.')
    parser.add_argument('--model', type=str, default='net1', choices=('net0', 'net1'), help='choose a network model')
    parser.add_argument('--n_class', type=int, default=64, help='number of classes each level.')
    parser.add_argument('--data_path', type=str, default='./datasets', help='path to the dataset.')
    parser.add_argument('--training_info', type=str, default='train_20f.txt', help='the file that contains the list of training images.')
    parser.add_argument('--n_iter', type=int, default=30000)
    parser.add_argument('--meta_lr', type=float, default=5e-4, help='learning rate.')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate.')
    parser.add_argument('--k_step', type=int, default=2, help='k-step SGD.')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size is fixed to 1.')
    parser.add_argument('--aug', type=str2bool, default=True) # useless?
    parser.add_argument('--train_id', type=str, default='', help='an identifier of the experiment.')
    parser.add_argument('--log-summary', type=str, default='progress_log_summary.txt', metavar='PATH', help='.txt file to save per-epoch stats.')
    args = parser.parse_args()

    args.scenes = [
    'apt1/kitchen',
    'apt1/living',
    'apt2/bed',
    'apt2/kitchen',
    'apt2/living',
    'apt2/luke',
    'office1/gates362',
    #'office1/gates381',
    'office1/lounge',
    'office1/manolis',
    'office2/5a',
    'office2/5b'
    ]

    for scene in args.scenes:
        cmd = 'python partition.py --data_path {} --dataset 12S --scene {} --training_info {} --n_class {}'.format(
            args.data_path,
            scene,
            args.training_info,
            args.n_class
            )
        os.system(cmd)

    seed = 0
    setup_seed(seed)
    train(args)

