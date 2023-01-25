import os
from cv2 import trace
import yaml
import pickle
import random
import argparse
from argparse import ArgumentParser
from omegaconf import OmegaConf

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP
import torch.optim as optim

import wandb

# 必要なモジュールをimport
import dataset.wood_dataset as wood_dataset
from dataset.wood_dataset import WoodDataset

from backbone.resnet import *
from patch_descriptor.patch_descriptor import *

from memory_bank.get_memory_bank import *

from trainer.loss import *
from trainer.trainer import *

from utils.metric import *
from utils.visualizer import * 



def argparser():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/config.yaml')
    
    parser.add_argument('--tr_batch_size', type=int, default=4)
    parser.add_argument('--tt_batch_size', type=int, default=4)
    parser.add_argument('--emb_batch_size', type=int, default=4)
    
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    
    return parser


def main(config_path, tr_batch_size, tt_batch_size, emb_batch_size, device_id, num_workers):
    
    # configを読み込む
    cfg = OmegaConf.load(config_path)
    
    # ----------------dataset & dataloaderを定義-----------------
    data_info = cfg.data
    tr_dataset = WoodDataset(csv_path=data_info.train_csv_path,
                             resize=(data_info.resize.h, data_info.resize.w),
                             crop_flag=data_info.crop.flag,
                             cropsize=(data_info.crop.h, data_info.crop.w),
                             is_train=True)
    tt_dataset = WoodDataset(csv_path=data_info.test_csv_path,
                             resize=(data_info.resize.h, data_info.resize.w),
                             crop_flag=data_info.crop.flag,
                             cropsize=(data_info.crop.h, data_info.crop.w),
                             is_train=False)
    emb_dataset = WoodDataset(csv_path=data_info.train_csv_path,
                              resize=(data_info.resize.h, data_info.resize.w),
                              crop_flag=data_info.crop.flag,
                              cropsize=(data_info.crop.h, data_info.crop.w),
                              is_train=True)
    tr_loader = DataLoader(dataset=tr_dataset,
                           batch_size=tr_batch_size,
                           pin_memory=True,
                           shuffle=True,
                           num_workers=num_workers)
    tt_loader = DataLoader(dataset=tt_dataset,
                           batch_size=tt_batch_size,
                           pin_memory=True,
                           shuffle=False,
                           num_workers=num_workers)
    emb_loader = DataLoader(dataset=tr_dataset,
                            batch_size=emb_batch_size,
                            pin_memory=True,
                            shuffle=False,
                            num_workers=num_workers)
    
    # ----------------------feature_extractorを定義-------------------------
    model_info = cfg.model
    if model_info.backbone == 'wide_resnet50_2':
        feature_extractor = wide_resnet50_2(pretrained=True, progress=True)
        dim = 1792
    elif model_info.backbone == 'resnet18':
        feature_extractor = resnet18(pretrained=True, progress=True)
        dim = 448
    # deviceの設定を行う
    device = torch.device(f'cuda:{device_id}')
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # --------------------patch_descriptorを定義----------------------
    patch_discriptor_info = cfg.cfa.patch_discriptor
    patch_descriptor = PatchDescriptor(dim=dim, gamma_d=patch_discriptor_info.gamma_d)

    # --------------------圧縮したmemory_bankを定義----------------------
    memory_bank_compression_info = cfg.cfa.memory_bank_compression
    memory_bank = GetMemoryBank(emb_loader=emb_loader,
                                feature_extractor=feature_extractor,
                                patch_descriptor=patch_descriptor,
                                device=device,
                                method=memory_bank_compression_info.method,
                                scale_for_cluster=memory_bank_compression_info.scale_for_cluster**2).memory_bank
   
    # -----------------------------lossを定義----------------------------
    loss_info = cfg.cfa.loss
    loss = Loss(memory_bank=memory_bank,
                K=loss_info.loss_att.K,
                r=loss_info.loss_att.r,
                J=loss_info.loss_rep.J,
                alpha=loss_info.loss_rep.alpha,
                nu=loss_info.nu)
    
    # -------------------------optimizerを定義---------------------------
    optimizer_info = cfg.optimizer
    params = [{'params': patch_descriptor.parameters()},
              {'params': loss.parameters()}]
    if optimizer_info.method == 'adam':
        optimizer = optim.Adam(params=params, 
                               lr=optimizer_info.adam.lr,
                               betas=(optimizer_info.adam.beta1, optimizer_info.adam.beta2),
                               weight_decay=optimizer_info.adam.weight_decay,
                               )
    elif optimizer_info.method == 'adamw':
        optimizer = optim.AdamW(params=params, 
                                lr=optimizer_info.adamw.lr,
                                betas=(optimizer_info.adamw.beta1, optimizer_info.adamw.beta2),
                                weight_decay=optimizer_info.adamw.weight_decay,
                                )
    elif optimizer_info.method == 'sgd':
        optimizer = optim.SGD(params=params,
                              weight_decay=optimizer_info.sgd.weight_decay)
    else:
        raise ValueError("optimizer is not defined")
    
    # ----------------------------wandbを定義-----------------------------
    wandb_name = input('wandb_name: ')
    wandb.init(project='Master Thesis', name=wandb_name)
    
    # ----------------------------trainerを定義----------------------------
    patch_descriptor.to(device)
    loss.to(device)
    trainer_info = cfg.trainer
    trainer = Trainer(tr_loader=tr_loader,
                      tt_loader=tt_loader,
                      feature_extractor=feature_extractor,
                      patch_descriptor=patch_descriptor,
                      memory_bank=memory_bank,
                      loss=loss,
                      optimizer=optimizer,
                      softmin_K=cfg.cfa.test.softmin_K,
                      device=device,
                      num_epoch=trainer_info.num_epoch,
                      wandb=wandb,
                      save_root_path=trainer_info.save_root_path)
    trainer.run()
        
    
if __name__=='__main__':
    parser = argparser()
    args = parser.parse_args()
    
    config_path = args.config_path
    tr_batch_size = args.tr_batch_size
    tt_batch_size = args.tt_batch_size 
    emb_batch_size = args.emb_batch_size
    device_id = args.device_id
    num_workers = args.num_workers 
    
    main(config_path, tr_batch_size, tt_batch_size, emb_batch_size, device_id, num_workers)
    
    