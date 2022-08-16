from __future__ import division

import sys, argparse
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models import get_model
from datasets import get_dataset
from loss import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from models.superpoint import SuperPoint


def train(args):
    
    # datasets.
    if args.dataset == '7S': 
        dataset = get_dataset('7S')
        dataset_val = get_dataset('7S_val')
    if args.dataset == 'Cambridge': 
        dataset = get_dataset('Cambridge')
    dataset = dataset(n_class=args.n_class, root=args.data_path, info=args.training_info, dataset=args.dataset, scene=args.scene, aug=args.aug)
    trainloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    if args.dataset == '7S':
        dataset_val = dataset_val(n_class=args.n_class, root=args.data_path, dataset=args.dataset, scene=args.scene, aug=False)
        trainloader_val = data.DataLoader(dataset_val, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # loss.
    cls_loss = CELoss()
    w1, w2 = 1, 1

    # network.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    extractor = SuperPoint()
    # for m in extractor.parameters():
    #     m.requires_grad = False
    # extractor.eval()
    extractor.to(device)

    classifier = get_model(args.model, args.n_class)
    classifier.init_weights()
    classifier.to(device)
    
    # optimizer.
    optimizer = torch.optim.Adam(
        [
        {'params':classifier.parameters()}, 
        {'params': extractor.parameters(), 'lr': 0.1*args.init_lr} 
        ], 
        lr=args.init_lr, eps=1e-8, betas=(0.9, 0.999))

    # output path.
    model_id = "{}-{}-{}-initlr{}-iters{}-bsize{}-aug{}-{}".format(\
                    args.dataset, args.scene.replace('/','.'),
                    args.model, args.init_lr, args.n_iter, args.batch_size, 
                    int(args.aug), args.train_id)
    save_path = Path(model_id)
    args.save_path = 'checkpoints'/save_path
    args.save_path.mkdir(parents=True, exist_ok=True)
    start_epoch = 1

    if args.init_weight is not None:
        print("Loading meta-learning pre-trained classifier model ")
        mapping = torch.load(args.init_weight, map_location=device)
        classifier.load_state_dict(mapping['model_state'])


    # start training...
    args.n_epoch = int(np.ceil(args.n_iter * args.batch_size / len(dataset)))

    for epoch in range(start_epoch, args.n_epoch+1):

        lr = args.init_lr

        extractor.train()
        classifier.train()
        train_loss_list = []
        lbl_1_loss_list = []
        lbl_2_loss_list = []
        lbl_1_loss_list_val = []
        lbl_2_loss_list_val = []
        cls_acc = []
        val_cls_acc = []
        for _, (img, mask, lbl_1, lbl_2, lbl_1_oh, lbl_2_oh) in enumerate(tqdm(trainloader)):

            if mask.sum() == 0:
                continue

            optimizer.zero_grad()

            img = img.to(device)
            mask = mask.to(device)

            lbl_1 = lbl_1.to(device)
            lbl_2 = lbl_2.to(device)
            lbl_1_oh = lbl_1_oh.to(device)
            lbl_2_oh = lbl_2_oh.to(device)
            feature_map = extractor(img)
            lbl_2_pred, lbl_1_pred = classifier(feature_map, lbl_1_oh, lbl_2_oh)
            lbl_1_loss = cls_loss(lbl_1_pred, lbl_1, mask)
            lbl_2_loss = cls_loss(lbl_2_pred, lbl_2, mask)
            train_loss = w1*lbl_1_loss + w2*lbl_2_loss

            # viz
            lbl_1_loss_list.append(lbl_1_loss.item())
            lbl_2_loss_list.append(lbl_2_loss.item())  
            # add cls accuracy
            lbl_1_p = torch.argmax(lbl_1_pred, dim=1)  
            lbl_2_p = torch.argmax(lbl_2_pred, dim=1)  
            lbl_p = (lbl_1_p * args.n_class + lbl_2_p)  
            lbl_gt = (lbl_1 * args.n_class + lbl_2)
            idx = torch.eq(lbl_p, lbl_gt)
            mask_new = mask.mul(idx)
            accuracy = torch.sum(mask_new)/ torch.sum(mask) 
            cls_acc.append(accuracy.item())

            train_loss_list.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", np.mean(train_loss_list), epoch)
        writer.add_scalar("Acc/train", np.mean(cls_acc), epoch)

        with open(args.save_path/args.log_summary, 'a') as logfile:
            logtt = 'train: Epoch {}/{} - lr: {} - cls_loss_1: {}' \
                    ' - cls_loss_2: {} - train_loss: {} - cls_acc: {} \n'.format(
                        epoch, args.n_epoch, lr, 
                        np.mean(lbl_1_loss_list), np.mean(lbl_2_loss_list),
                        np.mean(train_loss_list),np.mean(cls_acc))
            print(logtt)
            logfile.write(logtt)


        if args.dataset == '7S':
            if epoch%20==1:
                
                extractor.eval()
                classifier.eval()
                for _, (img_val, mask_val, lbl_1_val, lbl_2_val, lbl_1_oh_val, 
                        lbl_2_oh_val) in enumerate(tqdm(trainloader_val)):
               
                    if mask_val.sum() == 0:
                        continue

                    img_val = img_val.to(device)
                    mask_val = mask_val.to(device)
                    lbl_1_val = lbl_1_val.to(device)
                    lbl_2_val = lbl_2_val.to(device)
                    lbl_1_oh_val = lbl_1_oh_val.to(device)
                    lbl_2_oh_val = lbl_2_oh_val.to(device)
                    feature_map_val = extractor(img_val)
                    lbl_2_pred_val, lbl_1_pred_val = classifier(feature_map_val, lbl_1_oh_val, lbl_2_oh_val)
                    lbl_1_loss_val = cls_loss(lbl_1_pred_val, lbl_1_val, mask_val)
                    lbl_2_loss_val = cls_loss(lbl_2_pred_val, lbl_2_val, mask_val)
                    # train_loss_val = w1*lbl_1_loss_val + w2*lbl_2_loss_val

                    lbl_1_loss_list_val.append(lbl_1_loss_val.item())
                    lbl_2_loss_list_val.append(lbl_2_loss_val.item())  
                    # add cls accuracy
                    lbl_1_p_val = torch.argmax(lbl_1_pred_val, dim=1)  
                    lbl_2_p_val = torch.argmax(lbl_2_pred_val, dim=1)  
                    lbl_p_val = (lbl_1_p_val * args.n_class + lbl_2_p_val)  
                    lbl_gt_val = (lbl_1_val * args.n_class + lbl_2_val)
                    idx_val = torch.eq(lbl_p_val, lbl_gt_val)
                    mask_new_val = mask_val.mul(idx_val)
                    accuracy_val = torch.sum(mask_new_val)/ torch.sum(mask_val) 

                    val_cls_acc.append(accuracy_val.item())
                # writer.add_scalar("Loss/cls_1", np.mean(lbl_1_loss_list_val), epoch)
                # writer.add_scalar("Loss/cls_2", np.mean(lbl_2_loss_list_val), epoch)
                writer.add_scalar("Acc/val", np.mean(val_cls_acc), epoch)

                with open(args.save_path/args.log_summary, 'a') as logfile:
                    logtt = 'Validation: cls_loss_1_val: {}' \
                            ' - cls_loss_2_val: {}  - val_cls_acc: {} \n'.format(
                                np.mean(lbl_1_loss_list_val), np.mean(lbl_2_loss_list_val),  np.mean(val_cls_acc))

                    print(logtt)
                    logfile.write(logtt)

        if epoch % int(np.floor(args.n_epoch / 5.)) == 0:
            save_state(args.save_path, epoch, classifier, extractor, optimizer, suffix=epoch)
    writer.flush()
    save_state(args.save_path, epoch, classifier, extractor, optimizer)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train the SRC model to memorize the scene.')
    parser.add_argument('--model', type=str, default='net1', help='choose a network model')
    parser.add_argument('--n_class', type=int, default=64, help='number of classes each level. 7S: 64; Cambridge: 100')
    parser.add_argument('--init_weight', type=str, default=None, help='path to pre-trained network parameters.')
    parser.add_argument('--data_path', type=str, required=True, help='path to the dataset.')
    parser.add_argument('--dataset', type=str, default='7S', choices=('7S', 'Cambridge'), help='choose a dataset.')
    parser.add_argument('--scene', type=str, default='chess', help='choose a scene from the dataset.')
    parser.add_argument('--training_info', type=str, default='train_fewshot.txt', help='the file that contains the list of training images.')
    parser.add_argument('--n_iter', type=int, default=9000, help='number of training iterations. 7Scenes: 9k; Cambridge: 30k')
    parser.add_argument('--init_lr', type=float, default=5e-4, help='initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size is fixed to 1.')
    parser.add_argument('--aug', type=str2bool, default=True, help='w/ or w/o data augmentation.')
    parser.add_argument('--train_id', type=str, default='', help='an identifier of the experiment.')
    parser.add_argument('--log-summary', type=str, default='progress_log_summary.txt', metavar='PATH', help='.txt file to save per-epoch stats.')
    args = parser.parse_args()
    
    if args.dataset == '7S':
        if args.scene not in ['chess', 'heads', 'fire', 'office', 'pumpkin', 'redkitchen','stairs']:
            print('selected scene is not valid.')
            sys.exit()

    if args.dataset == 'Cambridge':
        if args.scene not in ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
            print('selected scene is not valid.')
            sys.exit()

    seed = 0
    setup_seed(seed)
    train(args)

