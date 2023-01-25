import torch 
import torch.nn as nn 
import torch.nn.functional as  F 

from tqdm import tqdm
import numpy as np 
from einops import rearrange
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.random_projection import SparseRandomProjection

from memory_bank.kcenter_greedy import KCenterGreedy


class GetMemoryBank(object):
    def __init__(self, emb_loader, feature_extractor, patch_descriptor, device, method, scale_for_cluster):
        self.emb_loader = emb_loader
        
        self.feature_extractor = feature_extractor
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        self.patch_descriptor = patch_descriptor.to(device)
        self.device = device
        
        self.method = method 
        self.scale_for_cluster = scale_for_cluster
        
        if method == 'centroid':
            memory_bank = self.centroid()
            memory_bank = rearrange(memory_bank, 'b c h w -> (b h w) c')
        elif method == 'mini_batch_kmeans':
            memory_bank = self.mini_batch_kmeans()
        elif method == 'kcenter_greedy':
            memory_bank = self.kcenter_greedy()
        
        memory_bank = torch.Tensor(memory_bank).to(device)
        memory_bank = memory_bank.detach()
        self.memory_bank = nn.Parameter(memory_bank, requires_grad=False)

    def get_phi_p(self, x):
        features = self.feature_extractor(x)
        embedding = None
        for feature in features:
            feature = self.feature_pooler(feature)
            if embedding is None:
                embedding = feature
            else:
                embedding = torch.cat((embedding, F.interpolate(feature, embedding.size(2), mode='bilinear')), dim=1)
        phi_p = self.patch_descriptor(embedding)
        return phi_p
    
    @torch.no_grad()
    def centroid(self):
        memory_bank = 0
        for i, (x, _, _,) in enumerate(tqdm(self.emb_loader, 'memory bank compression -->')):
            x = x.to(self.device)
            phi_p = self.get_phi_p(x)
            memory_bank = ((memory_bank * i) + torch.mean(phi_p, dim=0, keepdim=True)) / (i+1) 
            
        return memory_bank
    
    @torch.no_grad()
    def mini_batch_kmeans(self):
        for i, (x, _, _,) in enumerate(tqdm(self.emb_loader, 'memory bank compression -->')):
            x = x.to(self.device)
            phi_p = self.get_phi_p(x)
            if i == 0:
                n_clusters = int((self.scale**2))
                kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                         random_state=0,
                                         batch_size=self.scale_for_cluster)
            phi_p = rearrange(phi_p, 'b c h w -> (b h w) c').detach().cpu().numpy()
            kmeans = kmeans.partial_fit(phi_p)
        memory_bank = kmeans.cluster_centers_
        
        return memory_bank
    
    @torch.no_grad()
    def kcenter_greedy(self):
        total_embeddings = None
        for i, (x, _, _,) in enumerate(tqdm(self.emb_loader, 'memory bank compression -->')):
            x = x.to(self.device)
            phi_p = self.get_phi_p(x)
            if total_embeddings is None:
                total_embeddings =  phi_p.to('cpu')
            else:
                total_embeddings = torch.cat([total_embeddings, phi_p.to('cpu')], dim=0)
        
        total_embeddings = total_embeddings.reshape(-1, total_embeddings.size(1)).numpy()
        selector = KCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(N=self.scale_for_cluster)
        memory_bank = total_embeddings[selected_idx]
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', memory_bank.shape)
        
        return memory_bank