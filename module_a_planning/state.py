# module_a_planning/state.py
# STRIPS-style state representation for the Blocks World on a Pallet scenario.
# Schema fixed per §3.1 of the build spec — do not change field names or types.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class WorldState:
    # block -> supporting object ("PALLET" or another block id)
    on: Dict[str, str]
    holding: Optional[str] = None  # block id currently held by arm, or None

    def clear(self, block: str) -> bool:
        """True iff nothing is on top of block and block is not currently held."""
        return block != self.holding and not any(v == block for v in self.on.values())

    def arm_empty(self) -> bool:
        """True iff the arm holds nothing."""
        return self.holding is None

    def copy(self) -> "WorldState":
        return WorldState(on=dict(self.on), holding=self.holding)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorldState):
            return False
        return self.on == other.on and self.holding == other.holding

    def __hash__(self) -> int:
        return hash((frozenset(self.on.items()), self.holding))

    def __repr__(self) -> str:
        return f"WorldState(on={self.on}, holding={self.holding!r})"


# ---------------------------------------------------------------------------
# Action representation
# Actions are tuples: ("PICKUP", x), ("PUTDOWN", x), ("STACK", x, y), ("UNSTACK", x, y)
# ---------------------------------------------------------------------------

def _preconditions(action: tuple) -> List[str]:
    """Return a list of predicate strings that must hold for `action` to be applicable."""
    op = action[0]
    if op == "PICKUP":
        x = action[1]
        return [f"On({x},PALLET)", f"Clear({x})", "ArmEmpty"]
    elif op == "PUTDOWN":
        x = action[1]
        return [f"Holding({x})"]
    elif op == "STACK":
        x, y = action[1], action[2]
        return [f"Holding({x})", f"Clear({y})"]
    elif op == "UNSTACK":
        x, y = action[1], action[2]
        return [f"On({x},{y})", f"Clear({x})", "ArmEmpty"]
    else:
        raise ValueError(f"Unknown action: {action}")


def _check_preconditions(state: WorldState, action: tuple) -> bool:
    """Return True iff all preconditions of `action` hold in `state`."""
    for pred in _preconditions(action):
        if not _eval_pred(state, pred):
            return False
    return True


def _eval_pred(state: WorldState, pred: str) -> bool:
    """Evaluate a predicate string against a WorldState."""
    if pred == "ArmEmpty":
        return state.arm_empty()
    if pred.startswith("Holding("):
        x = pred[8:-1]
        return state.holding == x
    if pred.startswith("Clear("):
        x = pred[6:-1]
        return state.clear(x)
    if pred.startswith("On("):
        inner = pred[3:-1]
        x, y = inner.split(",", 1)
        return state.on.get(x) == y
    raise ValueError(f"Unknown predicate: {pred}")


def apply_action(state: WorldState, action: tuple) -> WorldState:
    """
    Pure function — returns a NEW WorldState after applying `action`.
    Raises ValueError if preconditions are not satisfied.
    """
    if not _check_preconditions(state, action):
        precs = _preconditions(action)
        failed = [p for p in precs if not _eval_pred(state, p)]
        raise ValueError(
            f"Preconditions not satisfied for {action}: failed={failed}, state={state}"
        )

    new_state = state.copy()
    op = action[0]

    if op == "PICKUP":
        x = action[1]
        del new_state.on[x]   # removes On(x, PALLET)
        new_state.holding = x

    elif op == "PUTDOWN":
        x = action[1]
        new_state.holding = None
        new_state.on[x] = "PALLET"

    elif op == "STACK":
        x, y = action[1], action[2]
        new_state.holding = None
        new_state.on[x] = y

    elif op == "UNSTACK":
        x, y = action[1], action[2]
        del new_state.on[x]   # removes On(x, y)
        new_state.holding = x

    return new_state


def is_goal_satisfied(state: WorldState, goal: dict) -> bool:
    """
    `goal` is a partial state dict with key "on" mapping blocks to supports.
    Returns True iff every listed On relation holds in `state`.
    """
    for block, support in goal.get("on", {}).items():
        if state.on.get(block) != support:
            return False
    return True
