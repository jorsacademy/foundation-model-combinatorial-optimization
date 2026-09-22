from __future__ import annotations
import itertools, numpy as np
from .data import distance_matrix,tour_length

def held_karp(coords):
    D=distance_matrix(coords); n=len(D)
    # DP[(mask,j)] = (cost, predecessor), start fixed at 0.
    dp={}
    for j in range(1,n):
        dp[(1<<(j-1),j)]=(D[0,j],0)
    for size in range(2,n):
        for subset in itertools.combinations(range(1,n),size):
            mask=sum(1<<(j-1) for j in subset)
            for j in subset:
                pmask=mask^(1<<(j-1))
                best=min((dp[(pmask,k)][0]+D[k,j],k) for k in subset if k!=j)
                dp[(mask,j)]=best
    full=(1<<(n-1))-1
    cost,last=min((dp[(full,j)][0]+D[j,0],j) for j in range(1,n))
    rev=[last]; mask=full; j=last
    while True:
        _,pred=dp[(mask,j)]
        mask ^= 1<<(j-1)
        if pred==0: break
        rev.append(pred); j=pred
    tour=(0,*reversed(rev))
    return tuple(tour),float(cost)

def brute_force_tsp(coords):
    n=len(coords); best=None
    for perm in itertools.permutations(range(1,n)):
        t=(0,*perm); val=tour_length(coords,t)
        if best is None or val<best[0]: best=(val,t)
    return best[1],best[0]
