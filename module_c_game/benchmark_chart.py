# module_c_game/benchmark_chart.py
# Benchmark all 3 algorithms at depths 2-6 from fixed start state.
# Saves nodes_explored.png per §5.2.5.

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_c_game.alphabeta import alphabeta  # noqa: E402
from module_c_game.game_state import START_STATE  # noqa: E402
from module_c_game.iterative_deepening import iterative_deepening  # noqa: E402
from module_c_game.minimax import minimax  # noqa: E402

DEPTHS = [2, 3, 4, 5, 6]
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nodes_explored.png"
)


def run_benchmark() -> dict:
    """Run all 3 algorithms at each depth. Returns dict of node lists."""
    mm_nodes = []
    ab_nodes = []
    ids_nodes = []

    for d in DEPTHS:
        print(f"  Depth {d}...", end=" ", flush=True)

        _, _, mm_n = minimax(START_STATE, d, True)
        _, _, ab_n = alphabeta(
            START_STATE, d, float("-inf"), float("inf"), True
        )
        _, _, stats = iterative_deepening(START_STATE, d)
        ids_n = sum(s["nodes_visited"] for s in stats)

        mm_nodes.append(mm_n)
        ab_nodes.append(ab_n)
        ids_nodes.append(ids_n)

        print(f"MM={mm_n:,}  AB={ab_n:,}  IDS={ids_n:,}")

    return {"minimax": mm_nodes, "alphabeta": ab_nodes, "ids": ids_nodes}


def save_chart(data: dict) -> None:
    """Plot and save the nodes_explored chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        DEPTHS, data["minimax"], "o-", color="#e74c3c",
        linewidth=2, markersize=7, label="MiniMax"
    )
    ax.plot(
        DEPTHS, data["alphabeta"], "s-", color="#3498db",
        linewidth=2, markersize=7, label="AlphaBeta"
    )
    ax.plot(
        DEPTHS, data["ids"], "^-", color="#2ecc71",
        linewidth=2, markersize=7, label="AlphaBeta+IDS"
    )

    ax.set_xlabel("Search Depth", fontsize=12)
    ax.set_ylabel("Nodes Visited", fontsize=12)
    ax.set_title("Dock Contention: Nodes Explored vs. Depth", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120)
    plt.close()
    print(f"Chart saved: {OUTPUT_PATH}")


def main():
    print("Running benchmark (depths 2-6)...")
    data = run_benchmark()
    save_chart(data)
    return data


if __name__ == "__main__":
    main()
