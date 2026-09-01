#!/usr/bin/env python
# module_a_planning/plan.py
# CLI entrypoint per §3.5:  python plan.py start.json goal.json
#
# Prints: the goal-stack plan, step-by-step execution trace,
#         the nonlinear plan layers, and any [REACTIVE] log lines.

from module_a_planning.state import WorldState, is_goal_satisfied
from module_a_planning.reactive import execute_with_reactive_layer
from module_a_planning.nonlinear_planner import plan_nonlinear
from module_a_planning.goal_stack_planner import plan_goal_stack
import json
import os
import sys

# Ensure repo root is on sys.path regardless of where the script is invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python plan.py <start.json> <goal.json> [disturbances.json]")
        sys.exit(1)

    start_path = sys.argv[1]
    goal_path = sys.argv[2]
    dist_path = sys.argv[3] if len(sys.argv) > 3 else None

    with open(start_path) as f:
        start_data = json.load(f)
    with open(goal_path) as f:
        goal_data = json.load(f)

    start_state = WorldState(**start_data)
    disturbances = []
    if dist_path:
        with open(dist_path) as f:
            disturbances = json.load(f)

    print("=" * 60)
    print("START STATE:", start_state)
    print("GOAL       :", goal_data)
    print("=" * 60)

    # --- Goal Stack Plan ---
    print("\n[GOAL-STACK PLANNER]")
    plan, trace = plan_goal_stack(start_state, goal_data)
    print("Plan:", plan)
    print("\nTrace:")
    for line in trace:
        print(" ", line)

    # --- Nonlinear Plan ---
    print("\n[NONLINEAR (POP) PLANNER]")
    pop_plan = plan_nonlinear(start_state, goal_data)
    linear_seq = pop_plan.linearize()
    layers = pop_plan.parallel_steps()
    print("Linearized plan:", linear_seq)
    print("Parallel layers:")
    for i, layer in enumerate(layers):
        print(f"  Layer {i+1}: {layer}")

    # --- Reactive Execution ---
    print("\n[REACTIVE EXECUTION]")
    result = execute_with_reactive_layer(plan, start_state, disturbances)
    print("Execution trace:")
    for line in result["trace"]:
        print(line)
    if result["reactive_log"]:
        print("\nReactive log:")
        for line in result["reactive_log"]:
            print(line)
    final = result["final_state"]
    satisfied = is_goal_satisfied(final, goal_data)
    print(f"\nFinal state: {final}")
    print(f"Goal satisfied: {satisfied}")


if __name__ == "__main__":
    main()
