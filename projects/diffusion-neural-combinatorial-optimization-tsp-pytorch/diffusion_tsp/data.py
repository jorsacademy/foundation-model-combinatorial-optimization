from __future__ import annotations
import numpy as np

def generate_tsp_batch(n_instances,*,seed=42,n_nodes=8):
    rng=np.random.default_rng(seed)
    return rng.uniform(0,1,size=(n_instances,n_nodes,2)).astype(np.float32)

def distance_matrix(coords):
    x=np.asarray(coords,float)
    diff=x[:,None,:]-x[None,:,:]
    return np.sqrt(np.sum(diff**2,axis=-1))

def tour_length(coords,tour):
    D=distance_matrix(coords); t=list(map(int,tour))
    return float(sum(D[t[i],t[(i+1)%len(t)]] for i in range(len(t))))

def tour_to_adjacency(tour,n):
    A=np.zeros((n,n),dtype=np.float32)
    t=list(map(int,tour))
    for i in range(n):
        a,b=t[i],t[(i+1)%n]
        A[a,b]=A[b,a]=1.0
    return A

def adjacency_signal(tour,n):
    A=tour_to_adjacency(tour,n)
    X=2*A-1
    np.fill_diagonal(X,0)
    return X.astype(np.float32)
