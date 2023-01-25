import torch 
import torch.nn as nn 

from patch_descriptor.coordconv import * 


class PatchDescriptor(nn.Module):
    def __init__(self, dim, gamma_d):
        super(PatchDescriptor, self).__init__()
        self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        
    def forward(self, p):
        phi_p = self.layer(p)
        
        return phi_p