# module_c_game/tests/test_game.py
# Unit tests for Module C (Game) - TC-C1 to TC-C4

from module_c_game.minimax import minimax
from module_c_game.iterative_deepening import iterative_deepening
from module_c_game.game_state import START_STATE
from module_c_game.dock_game import simulate_dock_game
from module_c_game.alphabeta import alphabeta
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# TC-C1 — Minimax vs AlphaBeta node counts
# ---------------------------------------------------------------------------
class TestTCC1:
    def test_alphabeta_prunes(self):
        """Alpha-beta should visit fewer (or equal) nodes than minimax at depth 4."""
        depth = 4
        _, _, mm_nodes = minimax(START_STATE, depth, True)
        _, _, ab_nodes = alphabeta(
            START_STATE, depth, float("-inf"), float("inf"), True)
        assert ab_nodes <= mm_nodes, (
            f"Alpha-beta ({ab_nodes}) visited more nodes than minimax ({mm_nodes})!"
        )

# ---------------------------------------------------------------------------
# TC-C2 — Minimax vs AlphaBeta values
# ---------------------------------------------------------------------------


class TestTCC2:
    def test_values_match(self):
        """Alpha-beta and minimax should return the exact same value for the same state and depth."""
        depth = 3
        mm_val, mm_move, _ = minimax(START_STATE, depth, True)
        ab_val, ab_move, _ = alphabeta(
            START_STATE, depth, float("-inf"), float("inf"), True)
        assert mm_val == ab_val, f"Values differ: MM={mm_val}, AB={ab_val}"

# ---------------------------------------------------------------------------
# TC-C3 — Iterative Deepening
# ---------------------------------------------------------------------------


class TestTCC3:
    def test_ids_runs(self):
        """IDS should return a valid move and value at max depth."""
        depth = 3
        val, move, stats = iterative_deepening(START_STATE, depth)
        assert move is not None, "IDS did not return a valid move"
        assert len(stats) == depth, "IDS did not return stats for each depth"
        assert stats[-1]["depth"] == depth, "Final depth stat mismatch"

# ---------------------------------------------------------------------------
# TC-C4 — Game Simulation
# ---------------------------------------------------------------------------


class TestTCC4:
    def test_simulate_game(self):
        """Game simulation should complete without errors and have a trace."""
        res = simulate_dock_game(depth=2, opponent="random")
        assert "winner" in res
        assert "plies" in res
        assert "trace" in res
        assert res["plies"] > 0
        assert len(res["trace"]) == res["plies"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
