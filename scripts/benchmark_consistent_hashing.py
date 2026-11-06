import json
from pathlib import Path
from typing import Dict
from irkit.core.sharding import ConsistentHashRing

def benchmark_consistent_hashing(num_keys: int = 10000):
    print("=" * 70)
    print("IRKit Consistent Hashing Ring: Node Rebalancing & Key Stability Test")
    print("=" * 70)

    # 1. Start with 3 shards
    initial_nodes = ["shard-0", "shard-1", "shard-2"]
    ring = ConsistentHashRing(nodes=initial_nodes, replicas=150)

    keys = [f"doc_key_{i}" for i in range(num_keys)]

    # Initial key assignment
    initial_mapping = {k: ring.get_node(k) for k in keys}

    initial_counts: Dict[str, int] = {}
    for node in initial_nodes:
        initial_counts[node] = sum(1 for n in initial_mapping.values() if n == node)

    print(f"\n[1/3] Initial 3-node distribution across {num_keys} keys:")
    for node, count in initial_counts.items():
        pct = (count / num_keys) * 100
        print(f"  • {node}: {count} keys ({pct:.1f}%)")

    # 2. Add a 4th node (scale out)
    new_node = "shard-3"
    print(f"\n[2/3] Adding node '{new_node}' (scaling from 3 to 4 nodes)...")
    ring.add_node(new_node)

    new_mapping = {k: ring.get_node(k) for k in keys}

    new_counts: Dict[str, int] = {}
    for node in initial_nodes + [new_node]:
        new_counts[node] = sum(1 for n in new_mapping.values() if n == node)

    print(f"New 4-node distribution:")
    for node, count in new_counts.items():
        pct = (count / num_keys) * 100
        print(f"  • {node}: {count} keys ({pct:.1f}%)")

    # 3. Analyze Key Movement
    moved_keys = sum(1 for k in keys if initial_mapping[k] != new_mapping[k])
    movement_pct = (moved_keys / num_keys) * 100
    theoretical_movement = (1.0 / 4.0) * 100

    # Verify key property of consistent hashing:
    # Keys only move from old nodes to the NEW node; no keys move between old nodes!
    displaced_to_other_old = sum(
        1 for k in keys if initial_mapping[k] != new_mapping[k] and new_mapping[k] != new_node
    )

    print("\n[3/3] Key Movement Analysis:")
    print(f"  • Total keys moved:                     {moved_keys} / {num_keys} ({movement_pct:.2f}%)")
    print(f"  • Theoretical optimal key migration:    {theoretical_movement:.2f}% (1/N = 1/4)")
    print(f"  • Deviation from theoretical optimum:   {abs(movement_pct - theoretical_movement):.2f}%")
    print(f"  • Keys moved between existing nodes:    {displaced_to_other_old} (0 expected in consistent hashing)")

    results = {
        "num_keys": num_keys,
        "initial_nodes": initial_nodes,
        "initial_counts": initial_counts,
        "added_node": new_node,
        "new_counts": new_counts,
        "keys_moved": moved_keys,
        "keys_moved_pct": round(movement_pct, 2),
        "theoretical_movement_pct": round(theoretical_movement, 2),
        "inter_node_migrations": displaced_to_other_old,
    }

    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / "consistent_hashing_results.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print(f"Raw results saved to: {report_file}")

if __name__ == "__main__":
    benchmark_consistent_hashing()

