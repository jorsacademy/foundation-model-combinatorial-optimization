from __future__ import annotations

from fmco import (
    DistillationConfig,
    UniversalRoutingPolicy,
    evaluate,
    exact_solution,
    generate_instance,
    summarize,
    train_multitask_student,
    train_task_teacher,
)


def build_examples(task: str, seeds: range) -> list[tuple[object, object]]:
    examples = []
    for seed in seeds:
        instance = generate_instance(
            task,
            customer_count=5,
            seed=seed,
            capacity=6.0,
        )
        examples.append((instance, exact_solution(instance)))
    return examples


def main() -> None:
    tsp = build_examples("tsp", range(4))
    cvrp = build_examples("cvrp", range(4))
    teachers = {
        "tsp": train_task_teacher(tsp, epochs=3),
        "cvrp": train_task_teacher(cvrp, epochs=3),
    }
    student, history = train_multitask_student(
        tsp + cvrp,
        teachers=teachers,
        model=UniversalRoutingPolicy(),
        config=DistillationConfig(epochs=4, teacher_weight=0.5),
        seed=7,
    )
    test = [
        generate_instance(
            task,
            customer_count=size,
            seed=100 + size,
            distribution=distribution,
            capacity=6.0,
        )
        for task in ("tsp", "cvrp")
        for size in (5, 6)
        for distribution in ("uniform", "clustered")
    ]
    rows = evaluate(student, test)
    print({"training_loss": history, "summary": summarize(rows)})


if __name__ == "__main__":
    main()
