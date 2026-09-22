import torch

from nco.heuristics import nearest_neighbor, two_opt
from nco.problems import generate_euclidean_tsp, is_valid_tour, tour_length


def test_square_tour_length_is_four() -> None:
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    tour = torch.tensor([[0, 1, 2, 3]])
    assert torch.allclose(tour_length(coords, tour), torch.tensor([4.0]))


def test_valid_tour_detection() -> None:
    tours = torch.tensor([[0, 2, 1, 3], [0, 1, 1, 3]])
    validity = is_valid_tour(tours)
    assert validity.tolist() == [True, False]


def test_instance_generation_shape_and_range() -> None:
    generator = torch.Generator().manual_seed(7)
    coords = generate_euclidean_tsp(3, 5, generator=generator)
    assert coords.shape == (3, 5, 2)
    assert torch.all((coords >= 0) & (coords < 1))


def test_nearest_neighbor_returns_permutations() -> None:
    coords = generate_euclidean_tsp(4, 8)
    tours = nearest_neighbor(coords)
    assert is_valid_tour(tours).all()


def test_two_opt_never_worsens_input() -> None:
    torch.manual_seed(3)
    coords = generate_euclidean_tsp(3, 8)
    initial = nearest_neighbor(coords)
    improved = two_opt(coords, initial, max_passes=10)
    assert is_valid_tour(improved).all()
    assert torch.all(tour_length(coords, improved) <= tour_length(coords, initial) + 1e-7)
