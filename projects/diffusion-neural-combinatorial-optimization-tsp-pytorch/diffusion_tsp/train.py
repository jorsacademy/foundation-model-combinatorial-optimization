from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from .data import adjacency_signal
from .diffusion import make_schedule,q_sample
from .exact import held_karp
from .model import EdgeDenoiser

@dataclass(frozen=True)
class TrainResult:
    model: EdgeDenoiser
    schedule: object
    history: tuple
    train_optimal_lengths: tuple

def build_exact_labels(coords_batch):
    signals=[]; lengths=[]
    for coords in np.asarray(coords_batch):
        tour,cost=held_karp(coords)
        signals.append(adjacency_signal(tour,len(coords)))
        lengths.append(cost)
    return np.stack(signals).astype(np.float32),tuple(float(x) for x in lengths)

def train_denoiser(
    train_coords,
    validation_coords,
    *,
    seed=42,
    steps=20,
    epochs=30,
    batch_size=32,
    hidden=64,
    lr=1e-3,
    device="cpu",
):
    np.random.seed(seed); torch.manual_seed(seed)
    train_x0,train_lengths=build_exact_labels(train_coords)
    val_x0,_=build_exact_labels(validation_coords)

    X=torch.tensor(train_coords,dtype=torch.float32)
    Y=torch.tensor(train_x0,dtype=torch.float32)
    XV=torch.tensor(validation_coords,dtype=torch.float32,device=device)
    YV=torch.tensor(val_x0,dtype=torch.float32,device=device)

    schedule=make_schedule(steps=steps,device=device)
    model=EdgeDenoiser(hidden=hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    rng=np.random.default_rng(seed)
    best=float("inf"); best_state=None; hist=[]

    for ep in range(1,epochs+1):
        model.train(); losses=[]
        order=rng.permutation(len(X))
        for st in range(0,len(order),batch_size):
            idx=order[st:st+batch_size]
            coords=X[idx].to(device); x0=Y[idx].to(device)
            t=torch.randint(0,steps,(len(idx),),device=device)
            xt,noise=q_sample(x0,t,schedule)
            pred=model(coords,xt,t)
            mask=(1-torch.eye(x0.shape[-1],device=device))[None]
            loss=((pred-noise)**2*mask).sum()/(mask.sum()*len(idx))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            gen=torch.Generator(device=device); gen.manual_seed(seed+ep)
            t=torch.full((len(XV),),steps//2,dtype=torch.long,device=device)
            noise=torch.randn(YV.shape,generator=gen,device=device)
            noise=.5*(noise+noise.transpose(-1,-2))
            eye=torch.eye(YV.shape[-1],device=device)
            noise=noise*(1-eye)
            xt,_=q_sample(YV,t,schedule,noise=noise)
            pred=model(XV,xt,t)
            mask=(1-eye)[None]
            val=float((((pred-noise)**2*mask).sum()/(mask.sum()*len(XV))).item())
        if val<best:
            best=val; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        hist.append((ep,float(np.mean(losses)),val))
    model.load_state_dict(best_state); model.eval()
    return TrainResult(model,schedule,tuple(hist),train_lengths)
