# dp\_scheduler Usage Guide

This document explains how to use the `dp_knapsack` function in `dp_scheduler.py` and how to run the provided test harness. It also includes a sample output example.

---

## 1. Prerequisites

- Python 3.11+
- `cleaned_data.csv` file containing at least the following columns:
  - `item_id`
  - `quantity`
  - `priority`
  - \##IMPORTANT : ensure it is in the folder /data
- `dp_scheduler.py` with the `dp_knapsack(items, n, capacity)` function defined.



## 2. Function Signature

```python
def dp_knapsack(items: List[Dict], n: int, capacity: int) -> Tuple[List[List[int]], List[int]]:
    """
    Solves the 0–1 knapsack problem using dynamic programming.

    Args:
      - items: a list of dictionaries, each with keys:
          - `id` (int): unique item identifier
          - `weight` (int): weight of the item
          - `value` (int): value or priority of the item
      - n: the number of items (len(items))
      - capacity: the maximum total weight

    Returns:
      - dp_table: a 2D list of size (n+1) x (capacity+1), where dp_table[i][w]
        is the maximum value achievable with the first i items and weight limit w.
      - selected_indices: a list of indices (0-based) of items included in the optimal solution.
    """
```

## 3. Running the Test Harness

A sample test script `test_dp.py` is provided to load the first N items from `cleaned_data.csv` and run the knapsack solver.

1. Place `test_dp.py`, `dp_scheduler.py`, and `cleaned_data.csv` in the same directory (e.g., `amine/data/`).
2. Update `test_dp.py` if you wish to change the number of items or capacity (default is 50 items and capacity 50).
3. From that directory, run:
   ```bash
   python test_dp.py
   ```

## 4. Sample `test_dp.py` Snippet

```python
from dp_scheduler import dp_knapsack

# Load items: list of dicts with id, weight, value
items = load_first_n_items("cleaned_data.csv", n=50)
n_items = len(items)
capacity = 50

# Call solver
dp_table, selected_ids = dp_knapsack(items, n_items, capacity)

total_priority = dp_table[n_items][capacity]
selected_item_ids = [ items[i]["id"] for i in selected_ids ]

print("Selected item IDs:", selected_item_ids)
print("Total items:      ", len(selected_item_ids))
print("Total weight:     ", sum(items[i]["weight"] for i in selected_ids))
print("Total priority:   ", total_priority)
```

## 5. Expected Output Example

```
Selected item IDs: [2, 5, 9, 12]
Total items:       4
Total weight:      47
Total priority:    31
```

*(Values will vary based on your dataset.)*
