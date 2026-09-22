import torch

from nco.problems import generate_clustered_tsp, generate_euclidean_tsp


def test_clustered_generation_shape_and_range() -> None:
    generator = torch.Generator().manual_seed(9)
    coords = generate_clustered_tsp(
        4,
        12,
        n_clusters=3,
        cluster_std=0.04,
        generator=generator,
    )
    assert coords.shape == (4, 12, 2)
    assert torch.all((coords >= 0) & (coords <= 1))


def test_clustered_generation_is_reproducible() -> None:
    first = generate_clustered_tsp(2, 10, generator=torch.Generator().manual_seed(5))
    second = generate_clustered_tsp(2, 10, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(first, second)


def test_uniform_and_clustered_generators_support_different_sizes() -> None:
    for n_nodes in (20, 30, 50):
        uniform = generate_euclidean_tsp(2, n_nodes)
        clustered = generate_clustered_tsp(2, n_nodes)
        assert uniform.shape == clustered.shape == (2, n_nodes, 2)


def test_clustered_generator_validates_parameters() -> None:
    try:
        generate_clustered_tsp(1, 5, n_clusters=0)
    except ValueError as exc:
        assert "n_clusters" in str(exc)
    else:
        raise AssertionError("expected ValueError for n_clusters=0")
