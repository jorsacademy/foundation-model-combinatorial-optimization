import torch

from fmco.decoding import greedy_decode
from fmco.model import UniversalRoutingPolicy
from fmco.problems import exact_solution, generate_instance
from fmco.training import DistillationConfig, train_multitask_student


def test_shared_policy_decodes_both_tasks() -> None:
    model = UniversalRoutingPolicy()
    for task in ("tsp", "cvrp"):
        instance = generate_instance(
            task,
            customer_count=5,
            seed=4,
            capacity=6.0,
        )
        solution = greedy_decode(model, instance)
        assert solution.task == task
        assert solution.cost > 0.0


def test_multitask_training_changes_parameters() -> None:
    examples = []
    for task in ("tsp", "cvrp"):
        for seed in (0, 1):
            instance = generate_instance(
                task,
                customer_count=4,
                seed=seed,
                capacity=6.0,
            )
            examples.append((instance, exact_solution(instance)))
    model = UniversalRoutingPolicy()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trained, history = train_multitask_student(
        examples,
        model=model,
        config=DistillationConfig(
            epochs=1,
            learning_rate=1e-3,
            teacher_weight=0.0,
        ),
    )
    assert len(history) == 1
    assert any(
        not torch.equal(before[key], value)
        for key, value in trained.state_dict().items()
    )
