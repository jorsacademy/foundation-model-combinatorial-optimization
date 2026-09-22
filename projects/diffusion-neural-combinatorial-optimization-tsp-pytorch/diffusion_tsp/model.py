from __future__ import annotations
import math, torch
from torch import nn

def time_embedding(t,dim):
    half=dim//2
    freq=torch.exp(torch.linspace(0,-math.log(10000),half,device=t.device))
    arg=t.float()[:,None]*freq[None,:]
    emb=torch.cat([torch.sin(arg),torch.cos(arg)],dim=1)
    if emb.shape[1]<dim: emb=torch.nn.functional.pad(emb,(0,dim-emb.shape[1]))
    return emb

class EdgeDenoiser(nn.Module):
    def __init__(self,hidden=64,time_dim=16):
        super().__init__(); self.time_dim=time_dim
        self.node=nn.Sequential(nn.Linear(2,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.edge=nn.Sequential(
            nn.Linear(2*hidden+3+time_dim,hidden),nn.ReLU(),
            nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1)
        )
    def forward(self,coords,xt,t):
        B,N,_=coords.shape
        h=self.node(coords)
        hi=h[:,:,None,:].expand(-1,-1,N,-1)
        hj=h[:,None,:,:].expand(-1,N,-1,-1)
        dist=torch.sqrt(torch.sum((coords[:,:,None,:]-coords[:,None,:,:])**2,-1)+1e-12)
        deg=xt.sum(-1)/(N-1)
        di=deg[:,:,None].expand(-1,-1,N)
        dj=deg[:,None,:].expand(-1,N,-1)
        te=time_embedding(t,self.time_dim)[:,None,None,:].expand(-1,N,N,-1)
        f=torch.cat([hi,hj,dist[...,None],xt[...,None],(di+dj)[...,None],te],-1)
        out=self.edge(f).squeeze(-1)
        out=.5*(out+out.transpose(1,2))
        eye=torch.eye(N,device=out.device,dtype=out.dtype)
        return out*(1-eye)
