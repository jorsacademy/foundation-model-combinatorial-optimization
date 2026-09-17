from __future__ import annotations

import torch

from fmco.model import UniversalRoutingPolicy
from fmco.problems import RoutingInstance, RoutingSolution, route_cost


def greedy_decode(
    model: UniversalRoutingPolicy,
    instance: RoutingInstance,
) -> RoutingSolution:
    model.eval()
    with torch.no_grad():
        logits = model(instance)
    if instance.task == "tsp":
        visited = torch.zeros(
            instance.node_count,
            dtype=torch.bool,
            device=logits.device,
        )
        current = 0
        visited[current] = True
        tour = [current]
        for _ in range(instance.node_count - 1):
            row = logits[current].masked_fill(visited, -1.0e9)
            nxt = int(torch.argmax(row).item())
            visited[nxt] = True
            tour.append(nxt)
            current = nxt
        routes = (tuple(tour),)
    else:
        unvisited = set(range(1, instance.node_count))
        built: list[tuple[int, ...]] = []
        while unvisited:
            route = [0]
            remaining = instance.capacity
            current = 0
            while True:
                feasible = [
                    j
                    for j in sorted(unvisited)
                    if instance.demands[j] <= remaining + 1e-12
                ]
                if not feasible:
                    break
                row = logits[current]
                nxt = max(
                    feasible,
                    key=lambda j: (float(row[j].cpu()), -j),
                )
                unvisited.remove(nxt)
                route.append(nxt)
                remaining -= float(instance.demands[nxt])
                current = nxt
            route.append(0)
            built.append(tuple(route))
        routes = tuple(built)
    cost = route_cost(instance, routes)
    return RoutingSolution(instance.task, routes, cost)
