# Warehouse Brain

This repository contains the `warehouse-brain` implementation, an AI-powered warehouse management system integrating classical AI, uncertainty handling, expert systems, game-playing algorithms, and connectionist models. The system is split into five functional modules and an integration layer.

## Module A: Planning
The planning module implements both Nilsson's Goal Stack Planning and a Partial Order Planner (POP) to arrange boxes in the warehouse. It features hierarchical task expansion for pallet loading and a reactive execution layer that repairs plans when environmental disturbances occur. The nonlinear planner is capable of generating parallel execution steps for independent subgoals.

## Module B: Uncertainty
The uncertainty module evaluates sensor readings using six different formalisms: nonmonotonic logic, naive Bayes, certainty factors, Bayesian networks (using pgmpy), Dempster-Shafer theory, and fuzzy logic. It processes noisy telemetry data like vibration, current, and temperature to determine if warehouse equipment is damaged. A comparative analysis tool highlights the disagreements between these methods on edge cases.

## Module C: Game Playing
The game module governs multi-agent competition for shared docking resources using adversarial search. It models the dock negotiation as a zero-sum game, implementing standard Minimax, Alpha-Beta pruning, and Iterative Deepening Search. The algorithms evaluate optimal paths and prune the search space to allow deeper lookahead within time constraints.

## Module D: Connectionist
The connectionist module features neural network models for pattern recognition and anomaly detection. A Hopfield network built from scratch in NumPy provides content-addressable memory for recalling visual warehouse symbols even when corrupted. Additionally, an RNN built with PyTorch processes temporal sequences of sensor telemetry to predict subtle, drifting anomalies that threshold-based systems miss.

## Module E: Expert System
The expert system provides diagnostic capabilities using a forward-chaining rule engine written in pure Python. It evaluates symptoms flagged by the uncertainty module and infers root causes (e.g., motor failure, structural damage). The system includes an interactive knowledge acquisition component and provides traces explaining its diagnostic reasoning.

## Integration
The integration layer generates synthetic shift logs encompassing sensor readings, physical disturbances, and dock requests. A central control loop dispatches these events to the appropriate modules, linking Module B's uncertainty evaluations to Module E's diagnostics when damage is suspected. The `main.py` entrypoint orchestrates this entire warehouse simulation workflow.
