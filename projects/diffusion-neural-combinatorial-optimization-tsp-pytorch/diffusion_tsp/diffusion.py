from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

def make_schedule(steps=20,beta_start=1e-4,beta_end=.08,device="cpu"):
    betas=torch.linspace(beta_start,beta_end,steps,device=device)
    alphas=1-betas
    bars=torch.cumprod(alphas,dim=0)
    return DiffusionSchedule(betas,alphas,bars)

def symmetric_noise(shape,device="cpu"):
    z=torch.randn(shape,device=device)
    z=.5*(z+z.transpose(-1,-2))
    eye=torch.eye(shape[-1],device=device,dtype=z.dtype)
    return z*(1-eye)

def q_sample(x0,t,schedule,noise=None):
    if noise is None: noise=symmetric_noise(x0.shape,x0.device)
    ab=schedule.alpha_bars[t].view(-1,1,1)
    xt=ab.sqrt()*x0+(1-ab).sqrt()*noise
    eye=torch.eye(x0.shape[-1],device=x0.device,dtype=x0.dtype)
    return xt*(1-eye),noise
