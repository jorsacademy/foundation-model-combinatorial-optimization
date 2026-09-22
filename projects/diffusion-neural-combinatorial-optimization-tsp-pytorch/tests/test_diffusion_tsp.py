import unittest, numpy as np, torch
from diffusion_tsp import *

class Tests(unittest.TestCase):
    def test_held_karp_matches_bruteforce(self):
        coords=generate_tsp_batch(2,seed=10,n_nodes=6)
        for c in coords:
            h=held_karp(c); b=brute_force_tsp(c)
            self.assertAlmostEqual(h[1],b[1],places=9)

    def test_optimal_adjacency_has_degree_two(self):
        c=generate_tsp_batch(1,seed=11,n_nodes=7)[0]
        t,_=held_karp(c); A=tour_to_adjacency(t,7)
        np.testing.assert_array_equal(A.sum(1),np.full(7,2.0))

    def test_q_sample_symmetric_zero_diagonal(self):
        c=generate_tsp_batch(2,seed=12,n_nodes=6)
        x=[]
        for cc in c:
            t,_=held_karp(cc); x.append(adjacency_signal(t,6))
        x=torch.tensor(np.stack(x))
        sched=make_schedule(5)
        xt,noise=q_sample(x,torch.tensor([0,4]),sched)
        self.assertTrue(torch.allclose(xt,xt.transpose(-1,-2)))
        self.assertTrue(torch.allclose(torch.diagonal(xt,dim1=-2,dim2=-1),torch.zeros(2,6)))

    def test_model_gradient(self):
        coords=torch.rand(3,6,2)
        xt=torch.rand(3,6,6); xt=.5*(xt+xt.transpose(-1,-2))
        m=EdgeDenoiser(hidden=32)
        out=m(coords,xt,torch.tensor([1,2,3]))
        loss=out.square().mean(); loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_train_and_sample_valid_tour(self):
        tr=generate_tsp_batch(10,seed=13,n_nodes=6)
        va=generate_tsp_batch(4,seed=14,n_nodes=6)
        r=train_denoiser(tr,va,seed=2,steps=6,epochs=2,batch_size=5,hidden=32)
        heat=sample_edge_heatmap(r,va[0],seed=3,n_samples=2)
        tour=greedy_heatmap_tour(heat)
        self.assertTrue(valid_tour(tour,6))
        opt_t,opt=held_karp(va[0])
        _,improved=two_opt(va[0],tour)
        self.assertGreaterEqual(improved,opt-1e-9)

    def test_two_opt_never_worsens(self):
        c=generate_tsp_batch(1,seed=15,n_nodes=8)[0]
        t=tuple(range(8)); before=tour_length(c,t)
        _,after=two_opt(c,t)
        self.assertLessEqual(after,before+1e-12)

if __name__=="__main__":unittest.main()
