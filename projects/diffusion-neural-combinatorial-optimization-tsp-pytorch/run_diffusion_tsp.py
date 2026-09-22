from __future__ import annotations
import argparse, numpy as np
from diffusion_tsp import generate_tsp_batch,train_denoiser,evaluate,held_karp,brute_force_tsp

def self_test():
    coords=generate_tsp_batch(1,seed=1,n_nodes=6)[0]
    hk=held_karp(coords); bf=brute_force_tsp(coords)
    assert abs(hk[1]-bf[1])<1e-9
    tr=generate_tsp_batch(8,seed=2,n_nodes=6)
    va=generate_tsp_batch(3,seed=3,n_nodes=6)
    result=train_denoiser(tr,va,seed=4,steps=6,epochs=2,batch_size=4,hidden=32)
    assert len(result.history)==2
    print("Diffusion TSP self-test: OK")

def main(a):
    train=generate_tsp_batch(a.train_instances,seed=a.seed,n_nodes=a.nodes)
    val=generate_tsp_batch(a.validation_instances,seed=a.seed+1_000_000,n_nodes=a.nodes)
    test=generate_tsp_batch(a.test_instances,seed=a.seed+2_000_000,n_nodes=a.nodes)
    result=train_denoiser(
        train,val,seed=a.seed,steps=a.steps,epochs=a.epochs,
        batch_size=a.batch_size,hidden=a.hidden,device=a.device
    )
    best=min(x[2] for x in result.history)
    print(f"best validation noise MSE={best:.6f}")
    rows=evaluate(result,test,seed=a.seed+3_000_000,n_diffusion_samples=a.samples,device=a.device)
    print(f"{'method':<30}{'mean length':>13}{'mean gap':>12}{'max gap':>11}")
    for r in rows:
        print(f"{r.method:<30}{r.mean_length:13.4f}{r.mean_gap_pct:11.3f}%{r.max_gap_pct:10.3f}%")

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--nodes",type=int,default=8)
    p.add_argument("--train-instances",type=int,default=80)
    p.add_argument("--validation-instances",type=int,default=20)
    p.add_argument("--test-instances",type=int,default=20)
    p.add_argument("--steps",type=int,default=12)
    p.add_argument("--epochs",type=int,default=18)
    p.add_argument("--batch-size",type=int,default=20)
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--samples",type=int,default=3)
    p.add_argument("--device",default="cpu")
    return p.parse_args()

if __name__=="__main__":
    a=parse(); self_test() if a.self_test else main(a)
