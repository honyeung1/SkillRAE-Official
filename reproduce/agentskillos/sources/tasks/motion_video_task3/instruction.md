I am an AI course instructor, and I want to create a comprehensive 8-puzzle demonstration combining mathematical analysis, problem solving, and visualization for my students.

Initial state: 7 2 4 / 5 _ 6 / 8 3 1, where _ is the blank. Goal state: 1 2 3 / 4 5 6 / 7 8 _.

Step 1 — Mathematical analysis: Compute the total state space size for the 8-puzzle, determine how many configurations are solvable, verify whether this specific initial state can reach the goal state, and compute the Manhattan distance heuristic for the initial state. Save to puzzle_analysis.json with: total_configurations, solvable_configurations, initial_state, goal_state, inversion_count, is_solvable, manhattan_distance, and tile_distances array showing each of the 8 numbered tiles' (1-8) Manhattan distance.

Step 2 — Find an optimal or near-optimal solution to the puzzle. Save the solution as puzzle_plan.txt with one move per line.

Step 3 — Create puzzle_animation.mp4 showing the 3 \times 3 grid with tiles sliding step-by-step according to your solution, with a move counter displayed, ending at the solved state.