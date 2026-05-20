# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "minigrid",
#   "tqdm",
# ]
# ///

"""
Single-process seed scanner for NGoalsEnv. Iterates seeds (strided by worker_stride
starting at worker_id) until num_seeds solvable seeds are found.

Uses BFS pathfinding instead of BabyAIBot (which hangs on most seeds).

Usage:
    python scripts/find_solvable_seeds.py --num_seeds=10 --worker_id=0 --worker_stride=1 --output_file=seeds.txt
"""

import os
import sys
from collections import deque

import fire

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _bfs_reachable_set(grid, start):
    """BFS from start; returns set of all reachable (x, y) positions."""
    w, h = grid.width, grid.height
    visited = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited or nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            cell = grid.get(nx, ny)
            if cell is not None and cell.type == "wall":
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))
    return visited


def _bfs_distance(grid, start, goal):
    """BFS shortest path distance. Returns -1 if unreachable."""
    if start == goal:
        return 0
    w, h = grid.width, grid.height
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) == goal:
                return dist + 1
            if (nx, ny) in visited or nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            cell = grid.get(nx, ny)
            if cell is not None and cell.type == "wall":
                continue
            visited.add((nx, ny))
            queue.append(((nx, ny), dist + 1))
    return -1


def _is_solvable(env_inner, max_total_steps):
    """Check if the agent can visit all goals in sequence within the step budget."""
    grid = env_inner.grid
    agent_pos = tuple(env_inner.agent_pos)
    targets = env_inner.task_targets

    goal_positions = {}
    w, h = grid.width, grid.height
    for x in range(w):
        for y in range(h):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_positions[cell.color] = (x, y)

    for color in targets:
        if color not in goal_positions:
            return False

    reachable = _bfs_reachable_set(grid, agent_pos)
    for pos in goal_positions.values():
        if pos not in reachable:
            return False

    total_dist = 0
    current = agent_pos
    for color in targets:
        goal = goal_positions[color]
        d = _bfs_distance(grid, current, goal)
        if d < 0:
            return False
        total_dist += d
        if total_dist > max_total_steps:
            return False
        current = goal

    return True


def find_solvable_seeds(
    num_seeds: int = 100,
    worker_id: int = 0,
    worker_stride: int = 1,
    max_total_steps: int = 500,
    size: int = 8,
    num_pairs: int = 3,
    num_walls: int = 4,
    output_file: str = "solvable_seeds.txt",
):
    from tqdm import tqdm
    from environments.ngoals import NGoalsEnv

    env_kwargs = dict(size=size, num_pairs=num_pairs, num_walls=num_walls, mode="test")
    env = NGoalsEnv(render_mode="rgb_array", **env_kwargs)

    solvable = []
    candidate = worker_id
    tried = 0

    pbar = tqdm(total=num_seeds, desc=f"Worker {worker_id}", position=worker_id)
    while len(solvable) < num_seeds:
        env.reset(seed=candidate)

        if _is_solvable(env, max_total_steps):
            solvable.append(candidate)
            pbar.update(1)

        candidate += worker_stride
        tried += 1

    pbar.close()
    env.close()

    with open(output_file, "w") as f:
        for s in solvable:
            f.write(f"{s}\n")

    print(f"Worker {worker_id}: found {len(solvable)} solvable seeds after trying {tried} candidates")


if __name__ == "__main__":
    fire.Fire(find_solvable_seeds)
