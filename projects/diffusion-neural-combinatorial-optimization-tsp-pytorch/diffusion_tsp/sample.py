from __future__ import annotations
import numpy as np
import torch

from .diffusion import symmetric_noise
from .data import distance_matrix,tour_length

@torch.no_grad()
def sample_edge_heatmap(result,coords,*,seed=0,n_samples=4,device="cpu"):
    model=result.model.to(device); schedule=result.schedule
    if schedule.betas.device.type != torch.device(device).type:
        from .diffusion import DiffusionSchedule
        schedule=DiffusionSchedule(
            schedule.betas.to(device),
            schedule.alphas.to(device),
            schedule.alpha_bars.to(device),
        )
    c=torch.tensor(np.asarray(coords)[None],dtype=torch.float32,device=device)
    N=c.shape[1]
    scores=[]
    torch.manual_seed(seed)
    for s in range(n_samples):
        xt=symmetric_noise((1,N,N),device=device)
        x0_est=None
        for ti in reversed(range(len(schedule.betas))):
            t=torch.full((1,),ti,dtype=torch.long,device=device)
            eps=model(c,xt,t)
            beta=schedule.betas[ti]
            alpha=schedule.alphas[ti]
            abar=schedule.alpha_bars[ti]
            x0_est=(xt-(1-abar).sqrt()*eps)/abar.sqrt()
            x0_est=torch.clamp(x0_est,-1,1)
            if ti>0:
                mean=(xt-beta/(1-abar).sqrt()*eps)/alpha.sqrt()
                noise=symmetric_noise(xt.shape,device=device)
                xt=mean+beta.sqrt()*noise
            else:
                xt=x0_est
            eye=torch.eye(N,device=device)
            xt=.5*(xt+xt.transpose(-1,-2))*(1-eye)
        scores.append(x0_est[0].cpu().numpy())
    heat=np.mean(scores,axis=0)
    heat=.5*(heat+heat.T)
    np.fill_diagonal(heat,-np.inf)
    return heat

def greedy_heatmap_tour(heatmap,start=0):
    H=np.asarray(heatmap,float); n=len(H)
    unvisited=set(range(n)); unvisited.remove(start)
    tour=[start]; cur=start
    while unvisited:
        nxt=max(unvisited,key=lambda j:(H[cur,j],-j))
        tour.append(int(nxt)); unvisited.remove(nxt); cur=nxt
    return tuple(tour)

def nearest_neighbor_tour(coords,start=0):
    D=distance_matrix(coords); n=len(D)
    unvisited=set(range(n)); unvisited.remove(start)
    tour=[start]; cur=start
    while unvisited:
        nxt=min(unvisited,key=lambda j:(D[cur,j],j))
        tour.append(int(nxt)); unvisited.remove(nxt); cur=nxt
    return tuple(tour)

def two_opt(coords,tour):
    t=list(tour); n=len(t); best=tour_length(coords,t)
    improved=True
    while improved:
        improved=False
        for i in range(1,n-1):
            for k in range(i+1,n):
                cand=t[:i]+list(reversed(t[i:k+1]))+t[k+1:]
                val=tour_length(coords,cand)
                if val<best-1e-12:
                    t=cand; best=val; improved=True
                    break
            if improved: break
    return tuple(t),float(best)

def valid_tour(tour,n):
    return len(tour)==n and set(map(int,tour))==set(range(n))
