# https://github.com/lukasruff/Deep-SVDD-PyTorch
# licenses/MIT

import numpy as np 
from einops import rearrange

import torch 
import torch.nn as nn 


class Loss(nn.Module):
    def __init__(self, memory_bank, K, J, r, alpha, nu):
        super(Loss, self).__init__()
        self.memory_bank = memory_bank
        self.K = K 
        self.J = J 
        self.r = nn.Parameter(r*torch.ones(1), requires_grad=True)
        self.alpha = alpha 
        self.nu = nu
        
    def forward(self, phi_p):
        phi_p = rearrange(phi_p, 'b c h w -> b (h w) c')
        dist = None 
        for i in range(phi_p.size(0)):
            if dist is None:
                dist = torch.cdist(phi_p[i], self.memory_bank).unsqueeze(0) 
            else :
                dist = torch.cat([dist, torch.cdist(phi_p[i], self.memory_bank).unsqueeze(0)], dim=0)
        dist = dist ** 2

        dist = dist.topk(self.K+self.J, largest=False).values
        
        # Loss_attを計算する（上位Kの近傍が半径rの超球に収まるように）
        scores = dist[:,:, :self.K] - self.r**2
        loss_att = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        
        # Loss_repを計算する（上位(K+1~J)の近傍がr-alpha離れるように）
        scores = self.r**2 - dist[:,:, self.J:] - self.alpha
        loss_rep = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        
        loss = loss_att + loss_rep
        
        return loss