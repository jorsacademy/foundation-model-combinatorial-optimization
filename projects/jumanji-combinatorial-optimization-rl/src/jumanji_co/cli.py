from .knapsack import rollout_density_policy


def main() -> None:
    result = rollout_density_policy()
    print(f"selected_items={len(result.selected_items)}")
    print(f"total_reward={result.total_reward:.6f}")
    print(f"total_weight={result.total_weight:.6f}")
    print(f"remaining_budget={result.remaining_budget:.6f}")


if __name__ == "__main__":
    main()
