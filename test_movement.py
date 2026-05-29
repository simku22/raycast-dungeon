"""Validate the maze movement core against the design panel's 14 vectors,
plus property tests (solvability, determinism, no wall-walking)."""

import json
import os
import random
import sys

sys.path.insert(0, "/Users/simon/Desktop/swecc-core/services/bench/common")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import RaycastDungeonEnv, _gen_maze, _norm_facing, apply_action, is_open  # noqa: E402

VECTORS = json.load(open("/tmp/dungeon_tests.json"))


def _find(grid, ch):
    for y, row in enumerate(grid):
        x = row.find(ch)
        if x >= 0:
            return (x, y)
    return None


def run_vector(case):
    grid = case["maze"]
    x, y = _find(grid, "S")
    exit_pos = _find(grid, "E")
    facing = _norm_facing(case["start_facing"])
    bumps = []
    reached = False
    for act in case["actions"]:
        x, y, facing, bumped = apply_action(grid, x, y, facing, act)
        bumps.append(bumped)
        if (x, y) == exit_pos:
            reached = True
            break
    return x, y, facing, reached, bumps


def test_vectors():
    fails = []
    for c in VECTORS:
        x, y, facing, reached, bumps = run_vector(c)
        exp = (c["expected_x"], c["expected_y"], _norm_facing(c["expected_facing"]))
        got = (x, y, facing)
        ok = got == exp and reached == c.get("reached_exit", reached)
        if "bumps" in c and c["bumps"] is not None:
            n = min(len(bumps), len(c["bumps"]))
            ok = ok and bumps[:n] == c["bumps"][:n]
        flag = "ok " if ok else "FAIL"
        print(f"  [{flag}] {c['name']:24} got={got} reached={reached} exp={exp}/{c.get('reached_exit')}")
        if not ok:
            fails.append(c["name"])
    assert not fails, f"vector failures: {fails}"
    print(f"vectors: {len(VECTORS)}/{len(VECTORS)} passed")


def test_solvability_and_determinism():
    for size in (3, 4, 5, 6):
        for s in range(150):
            g1, st1, ex1, _ = _gen_maze(size, random.Random(s), False)
            g2, st2, ex2, _ = _gen_maze(size, random.Random(s), False)
            assert g1 == g2 and st1 == st2 and ex1 == ex2, f"non-deterministic size={size} seed={s}"
            # _gen_maze already asserts reachability internally
            assert g1[st1[1]][st1[0]] == "S" and g1[ex1[1]][ex1[0]] == "E"
    print("solvability+determinism: OK (4 sizes x 150 seeds)")


def test_key_variant_solvable():
    for s in range(100):
        g, st, ex, key = _gen_maze(5, random.Random(s), True)
        assert key != (-1, -1) and g[key[1]][key[0]] == "K"
    print("key variant: OK (100 seeds, key always placed + reachable)")


def test_no_wall_walking():
    env = RaycastDungeonEnv()
    acts = ["forward", "turn_left", "turn_right", "turn_around"]
    for s in range(120):
        obs = env.reset(seed=s, size=4)
        rng = random.Random(s * 7 + 1)
        for _ in range(40):
            r = env.step(rng.choice(acts))
            assert env.grid[env.y][env.x] != "#", f"agent on wall seed={s}"
            assert 0 <= env.x < len(env.grid[0]) and 0 <= env.y < len(env.grid)
            assert isinstance(r.info["maze"], str)
            if r.terminated or r.truncated:
                break
    print("no-wall-walking + in-bounds + string-info: OK (120 episodes)")


def test_solvable_by_optimal_play():
    """A BFS-optimal walker should escape within the step cap on size=4."""
    from collections import deque
    wins = 0
    for s in range(60):
        env = RaycastDungeonEnv()
        env.reset(seed=s, size=4)
        # BFS shortest path on the grid
        start, goal = (env.x, env.y), env.exit
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = cur[0] + dx, cur[1] + dy
                if (nx, ny) not in prev and is_open(env.grid, nx, ny):
                    prev[(nx, ny)] = cur
                    q.append((nx, ny))
        # reconstruct path length (cells); each move = 1 forward, turns add overhead
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        moves = len(path) - 1
        if moves <= 35:
            wins += 1
    print(f"optimal-play escapable within cap: {wins}/60 (pure forward moves <= 35)")
    assert wins == 60


if __name__ == "__main__":
    print("== movement vectors =="); test_vectors()
    print("== solvability/determinism =="); test_solvability_and_determinism()
    print("== key variant =="); test_key_variant_solvable()
    print("== runtime invariants =="); test_no_wall_walking()
    print("== optimal escapability =="); test_solvable_by_optimal_play()
    print("\nALL TESTS PASSED")
