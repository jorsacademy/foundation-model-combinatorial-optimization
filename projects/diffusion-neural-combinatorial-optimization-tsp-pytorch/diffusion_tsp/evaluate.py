from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .data import tour_length
from .exact import held_karp
from .sample import sample_edge_heatmap,greedy_heatmap_tour,nearest_neighbor_tour,two_opt

@dataclass(frozen=True)
class EvalRow:
    method:str
    mean_length:float
    mean_gap_pct:float
    max_gap_pct:float

def evaluate(result,coords_batch,*,seed=0,n_diffusion_samples=4,device="cpu"):
    rec={"Diffusion heatmap + 2-opt":[],"Nearest neighbor + 2-opt":[]}
    for i,coords in enumerate(np.asarray(coords_batch)):
        _,opt=held_karp(coords)
        heat=sample_edge_heatmap(result,coords,seed=seed+1009*i,n_samples=n_diffusion_samples,device=device)
        dt=greedy_heatmap_tour(heat); _,dl=two_opt(coords,dt)
        nt=nearest_neighbor_tour(coords); _,nl=two_opt(coords,nt)
        rec["Diffusion heatmap + 2-opt"].append((dl,100*(dl-opt)/opt))
        rec["Nearest neighbor + 2-opt"].append((nl,100*(nl-opt)/opt))
    out=[]
    for name,rows in rec.items():
        a=np.asarray(rows,float)
        out.append(EvalRow(name,float(a[:,0].mean()),float(a[:,1].mean()),float(a[:,1].max())))
    return tuple(out)
