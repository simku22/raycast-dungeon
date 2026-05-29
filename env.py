"""
Raycast Dungeon — a first-person maze-escape benchmark ("can an LLM play DOOM?").

The agent perceives the maze ONLY in first person (what's open ahead/left/right,
how far it can see down each corridor, a compass bearing to the exit) and issues
one discrete action per turn: forward / turn_left / turn_right / turn_around.
It must reach the exit before the step cap. LLMs are notoriously bad at building
a spatial map from egocentric text, so they wander, loop, and faceplant walls.

Coordinate convention (ONE convention everywhere):
    x = column (0 = left, +x = East), y = row (0 = top, +y = South).
    N=(0,-1)  E=(+1,0)  S=(0,+1)  W=(-1,0)
    turn_right = clockwise  N->E->S->W->N
    turn_left  = counter-cw N->W->S->E->N
The maze is a flat char grid: '#'=wall, '.'/'S'/'E'/'K'=floor. The agent walks
one char cell per `forward`. A `forward` into a wall is a no-op "bump".
"""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

# ── Facing tables ─────────────────────────────────────────────────────────────
ORDER = ["N", "E", "S", "W"]                       # clockwise
VEC = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
_FULL = {"north": "N", "east": "E", "south": "S", "west": "W"}


def _norm_facing(f: str) -> str:
    f = str(f).strip()
    return _FULL.get(f.lower(), f.upper()[:1] if f else "N")


def turn(facing: str, action: str) -> str:
    i = ORDER.index(facing)
    if action == "turn_right":
        return ORDER[(i + 1) % 4]
    if action == "turn_left":
        return ORDER[(i - 1) % 4]
    if action == "turn_around":
        return ORDER[(i + 2) % 4]
    return facing


def is_open(grid: list[str], x: int, y: int) -> bool:
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] != "#"


def apply_action(grid: list[str], x: int, y: int, facing: str, action: str):
    """Pure movement core (shared by env.step and the test suite).

    Returns (x, y, facing, bumped). Turns never bump; forward into a wall or
    out of bounds is a no-op bump.
    """
    if action in ("turn_left", "turn_right", "turn_around"):
        return x, y, turn(facing, action), False
    if action == "forward":
        dx, dy = VEC[facing]
        nx, ny = x + dx, y + dy
        if is_open(grid, nx, ny):
            return nx, ny, facing, False
        return x, y, facing, True          # bump
    return x, y, facing, False             # unknown action: stall


def _bearing(dx: int, dy: int) -> str:
    if dx == 0 and dy == 0:
        return "here"
    ns = "N" if dy < 0 else ("S" if dy > 0 else "")
    ew = "E" if dx > 0 else ("W" if dx < 0 else "")
    return ns + ew


def _gen_maze(size: int, rng: random.Random, require_key: bool):
    """Randomized-DFS (recursive backtracker) → fully connected spanning tree,
    rendered to a (2*size+1) char grid. Connectivity guarantees solvability."""
    W = 2 * size + 1
    grid = [["#"] * W for _ in range(W)]
    visited = [[False] * size for _ in range(size)]

    def c2g(cx, cy):
        return 2 * cx + 1, 2 * cy + 1

    stack = [(0, 0)]
    visited[0][0] = True
    while stack:
        cx, cy = stack[-1]
        gx, gy = c2g(cx, cy)
        grid[gy][gx] = "."
        nbrs = []
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):   # N,E,S,W
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < size and 0 <= ny < size and not visited[ny][nx]:
                nbrs.append((nx, ny, dx, dy))
        if nbrs:
            nx, ny, dx, dy = rng.choice(nbrs)
            visited[ny][nx] = True
            grid[gy + dy][gx + dx] = "."        # carve passage between cells
            ngx, ngy = c2g(nx, ny)
            grid[ngy][ngx] = "."
            stack.append((nx, ny))
        else:
            stack.pop()

    sx, sy = c2g(0, 0)
    ex, ey = c2g(size - 1, size - 1)
    grid[sy][sx] = "S"
    grid[ey][ex] = "E"

    kx = ky = -1
    if require_key:
        # place key on a random interior cell that isn't start or exit
        cells = [c2g(cx, cy) for cx in range(size) for cy in range(size)]
        choices = [(gx, gy) for gx, gy in cells if (gx, gy) not in ((sx, sy), (ex, ey))]
        kx, ky = rng.choice(choices)
        grid[ky][kx] = "K"

    rows = ["".join(r) for r in grid]
    # Runtime solvability assertion (belt-and-suspenders over the construction guarantee).
    assert _reachable(rows, (sx, sy), (ex, ey)), "generated maze is unsolvable"
    if require_key:
        assert _reachable(rows, (sx, sy), (kx, ky)), "key unreachable"
    return rows, (sx, sy), (ex, ey), (kx, ky)


def _reachable(grid: list[str], start, goal) -> bool:
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in VEC.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) not in seen and is_open(grid, nx, ny):
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


class RaycastDungeonEnv(BaseEnv):
    """First-person maze escape. One action per turn; reach the exit to win."""

    def __init__(self) -> None:
        self.size = 4
        self.require_key = False
        self.max_steps = 35
        self._rng = random.Random()

    # ── Episode start ──────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._rng.seed(seed)
        self.seed = 0 if seed is None else int(seed)
        self.size = int(params.get("size", self.size))
        self.require_key = bool(params.get("require_key", self.require_key))
        self.max_steps = int(params.get("max_steps", 35))

        self.grid, self.start, self.exit, self.key = _gen_maze(
            self.size, self._rng, self.require_key
        )
        self.x, self.y = self.start
        self.facing = "S"                      # face into the maze
        self.steps = 0
        self.bumps = 0
        self.has_key = not self.require_key
        self.got_key = False
        self.won = False
        self.last_action = "reset"
        self._prev = (self.x, self.y, self.facing)
        return self._observe(bumped=False)

    # ── One agent turn ───────────────────────────────────────────────────────────
    def step(self, action: Any) -> StepResult:
        act = str(action).strip()
        self._prev = (self.x, self.y, self.facing)
        self.steps += 1
        self.last_action = act

        self.x, self.y, self.facing, bumped = apply_action(
            self.grid, self.x, self.y, self.facing, act
        )

        reward = -0.02
        if bumped:
            self.bumps += 1
            reward -= 0.05

        # key pickup on entry
        if self.require_key and not self.has_key and (self.x, self.y) == self.key:
            self.has_key = True
            self.got_key = True
            reward += 0.15

        # win check
        at_exit = (self.x, self.y) == self.exit
        if at_exit and self.has_key:
            self.won = True
            reward += 1.0

        terminated = self.won
        truncated = (self.steps >= self.max_steps) and not terminated

        return StepResult(
            observation=self._observe(bumped),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=self._info(bumped),
        )

    # ── First-person observation (no full map!) ──────────────────────────────────
    def _observe(self, bumped: bool) -> dict[str, Any]:
        f = self.facing
        left, right, behind = turn(f, "turn_left"), turn(f, "turn_right"), turn(f, "turn_around")

        def look(d: str) -> str:
            dx, dy = VEC[d]
            return "open" if is_open(self.grid, self.x + dx, self.y + dy) else "wall"

        def depth(d: str) -> int:
            dx, dy = VEC[d]
            n, cx, cy = 0, self.x + dx, self.y + dy
            cap = 2 * self.size + 1
            while is_open(self.grid, cx, cy) and n < cap:
                n += 1
                cx += dx
                cy += dy
            return n

        ex, ey = self.exit
        on = "exit" if (self.x, self.y) == self.exit else (
            "key" if (self.require_key and (self.x, self.y) == self.key and not self.has_key) else "none"
        )
        return {
            "ahead": look(f),
            "left": look(left),
            "right": look(right),
            "behind": look(behind),
            "dist_ahead": depth(f),
            "dist_left": depth(left),
            "dist_right": depth(right),
            "facing": f,
            "exit_bearing": _bearing(ex - self.x, ey - self.y),
            "exit_distance": abs(ex - self.x) + abs(ey - self.y),
            "inventory": "key" if (self.require_key and self.has_key) else "none",
            "on_item": on,
            "steps_left": max(0, self.max_steps - self.steps),
            "last_action_bumped": "true" if bumped else "false",
        }

    # ── Trace/showcase metadata (strings only; agent never sees this) ─────────────
    def _info(self, bumped: bool) -> dict[str, str]:
        wall01 = [[1 if c == "#" else 0 for c in row] for row in self.grid]
        px, py, pf = self._prev
        return {
            "maze": json.dumps(wall01),
            "size": str(2 * self.size + 1),
            "x": str(self.x),
            "y": str(self.y),
            "facing": self.facing,
            "prev_x": str(px),
            "prev_y": str(py),
            "prev_facing": pf,
            "start_x": str(self.start[0]),
            "start_y": str(self.start[1]),
            "exit": json.dumps(list(self.exit)),
            "exit_x": str(self.exit[0]),
            "exit_y": str(self.exit[1]),
            "key_x": str(self.key[0]),
            "key_y": str(self.key[1]),
            "has_key": "true" if self.has_key else "false",
            "got_key_numeric": "1" if self.got_key else "0",
            "bump": "true" if bumped else "false",
            "bump_count": str(self.bumps),
            "won": "true" if self.won else "false",
            "won_numeric": "1" if self.won else "0",
            "steps_taken": str(self.steps),
            "last_action": self.last_action,
            "seed": str(self.seed),
        }
