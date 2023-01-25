import os 
import sys
import math
import numpy as np 
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm

import torch 
import torch.nn as nn 
import torch.nn.functional as  F 

sys.path.append('../utils')
from utils.metric import * 
from utils.visualizer import *


class Trainer:
    def __init__(self, 
                 tr_loader, tt_loader, 
                 feature_extractor, patch_descriptor,
                 memory_bank,
                 loss,
                 optimizer,
                 softmin_K,
                 device,
                 num_epoch,
                 wandb,
                 save_root_path
                ):
        # Dataloaderについて
        self.tr_loader = tr_loader
        self.tt_loader = tt_loader 
        # feature_extractor & patch_descriptorについて
        self.feature_extractor = feature_extractor
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        self.patch_descriptor = patch_descriptor
        # memory_bankについて
        self.memory_bank = memory_bank
        # 損失関数について
        self.loss = loss
        # 最適化関数について
        self.optimizer = optimizer
        # epochについて
        self.num_epoch = num_epoch
        # 推論時のsoftminを計算する数について
        self.softmin_K = softmin_K 
        # deviceについて
        self.device = device
        # wandbについて
        self.wandb = wandb 
        # 保存pathについて
        self.save_root_path = save_root_path
    
    def run(self):
        total_image_auroc = []
        total_pixel_auroc = []
        total_pixel_aupro = []

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_image_auroc = ax[0]
        fig_pixel_auroc = ax[1]
        
        best_image_auroc = -1
        best_pixel_auroc = -1
        best_pixel_aupro = -1
        
        torch.backends.cudnn.benchmark = True
        for epoch in tqdm(range(self.num_epoch)):
            # 学習を行う
            self.patch_descriptor.train()
            self.train(epoch)
            torch.cuda.empty_cache()
            
            # 推論を行う
            self.patch_descriptor.eval()
            (image_fpr, 
             image_tpr, 
             image_auroc, 
             pixel_fpr, 
             pixel_tpr, 
             pixel_auroc, 
             pixel_aupro
            ) = self.test(epoch)
            torch.cuda.empty_cache()
            
            # wandbのlogを出力
            self.wandb.log(
            {'image_roc_auc': image_auroc,
             'epoch': epoch+1}
            )
            self.wandb.log(
                {'pixel_roc_auc': pixel_auroc,
                 'epoch': epoch+1}
            )
            self.wandb.log(
                {'pixel_auc_pro': pixel_aupro,
                 'epoch': epoch+1}
            )
            
            # 結果をリストに格納する
            total_image_auroc.append(image_auroc)
            total_pixel_auroc.append(pixel_auroc)
            total_pixel_aupro.append(pixel_aupro)
            
            # 結果をplotする
            fig_image_auroc.plot(image_fpr, image_tpr, label='%s Image-AUROC: %.3f' % ('wood', image_auroc))
            fig_pixel_auroc.plot(pixel_fpr, pixel_tpr, label='%s Pixel-AUROC: %.3f' % ('wood', pixel_auroc))

            # patch_descriptorの重みを保存
            os.makedirs(os.path.join(self.save_root_path, 'weight'), exist_ok=True)
            torch.save(self.patch_descriptor.state_dict(),
                       os.path.join(self.save_root_path, 'weight', f'patch_descriptor-epoch={epoch+1}.pth'))

    def get_phi_p(self, x):
        features = self.feature_extractor(x)
        embedding = None
        for feature in features:
            feature = self.feature_pooler(feature)
            if embedding is None:
                embedding = feature
            else:
                embedding = torch.cat((embedding, F.interpolate(feature, embedding.size(2), mode='bilinear')), dim=1)
        self.h = embedding.size(2)
        phi_p = self.patch_descriptor(embedding)
        return phi_p
    
    def softmin_score(self, phi_p):
        phi_p = rearrange(phi_p, 'b c h w -> b (h w) c')
        dist = None 
        for i in range(phi_p.size(0)):
            if dist is None:
                dist = torch.cdist(phi_p[i], self.memory_bank).unsqueeze(0) 
            else :
                dist = torch.cat([dist, torch.cdist(phi_p[i], self.memory_bank).unsqueeze(0)], dim=0)

        dist = dist.topk(self.softmin_K, largest=False).values
    
        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)
        score = rearrange(dist, 'b (h w) c -> b c h w', h=self.h)
        return score
    
    @torch.enable_grad()
    def train(self, epoch):
        for (x, _, _) in self.tr_loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            phi_p = self.get_phi_p(x)
            loss = self.loss(phi_p)
            loss.backward()
            self.optimizer.step()
    
    @torch.no_grad()
    def test(self, epoch):
        image_list = list()
        gt_mask_list = list()
        gt_list = list()
        scores = None
        for x, y, mask in self.tt_loader:
            image_list.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            
            x = x.to(self.device)
            phi_p = self.get_phi_p(x)
            
            score = self.softmin_score(phi_p)
            score = score.cpu().detach()
            score = torch.mean(score, dim=1) 
            scores = torch.cat((scores, score), dim=0) if scores != None else score
        
        gt_mask = np.asarray(gt_mask_list)
        
        scores = upsample(scores, size=x.size(2), mode='bilinear')
        scores = gaussian_smooth(scores, sigma=4)
        scores = rescale(scores)
        
        r'Image Level AUROC'
        image_fpr, image_tpr, image_auroc = cal_img_roc(scores, gt_list)
        r'Pixel Level AUROC'
        pixel_fpr, pixel_tpr, pixel_auroc = cal_pxl_roc(gt_mask, scores)
        r'Pixel Level AUPRO'
        pixel_aupro = cal_pxl_pro(gt_mask, scores)
        
        return image_fpr, image_tpr, image_auroc, pixel_fpr, pixel_tpr, pixel_auroc, pixel_aupro
        