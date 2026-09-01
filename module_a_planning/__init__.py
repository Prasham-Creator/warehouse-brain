# module_a_planning/__init__.py
# Public API contract for Module A per §3.6.

from module_a_planning.goal_stack_planner import plan_goal_stack
from module_a_planning.hierarchical import expand_load_pallet
from module_a_planning.nonlinear_planner import plan_nonlinear
from module_a_planning.reactive import execute_with_reactive_layer
from module_a_planning.state import WorldState, apply_action, is_goal_satisfied

__all__ = [
    "WorldState",
    "apply_action",
    "is_goal_satisfied",
    "plan_goal_stack",
    "plan_nonlinear",
    "expand_load_pallet",
    "execute_with_reactive_layer",
]
