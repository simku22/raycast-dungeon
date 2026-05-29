"""Generate a SAMPLE replay bundle (real export schema) so the showcase can be
built/previewed before a live run. A deliberately-mediocre navigator policy
produces wandering, bumps, and the occasional escape. Replace later with:
    mesocosm run export RUN_ID -o showcase/data/replay.json
"""
from __future__ import annotations
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/Users/simon/Desktop/swecc-core/services/bench/common")
from env import RaycastDungeonEnv  # noqa

MODEL = "gemini/gemini-3.1-flash-lite"
ORDER = ["N", "E", "S", "W"]


from collections import deque  # noqa: E402
VEC = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}


def bfs_next_cell(env):
    """Next cell on a shortest path from the agent to the exit (sample only)."""
    start, goal = (env.x, env.y), env.exit
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            break
        for dx, dy in VEC.values():
            nx, ny = c[0] + dx, c[1] + dy
            if (nx, ny) not in prev and 0 <= ny < len(env.grid) and 0 <= nx < len(env.grid[0]) and env.grid[ny][nx] != "#":
                prev[(nx, ny)] = c
                q.append((nx, ny))
    path = []
    c = goal
    while c is not None and c in prev:
        path.append(c)
        c = prev[c]
    path.reverse()
    return path[1] if len(path) > 1 else goal


def policy(obs, rng, env):
    if rng.random() < 0.16:
        return rng.choice(["forward", "turn_left", "turn_right", "turn_around"])
    nxt = bfs_next_cell(env)
    want = (nxt[0] - env.x, nxt[1] - env.y)
    desired = next((f for f, v in VEC.items() if v == want), env.facing)
    order = ["N", "E", "S", "W"]
    diff = (order.index(desired) - order.index(env.facing)) % 4
    if diff == 0:
        return "forward"
    if diff == 1:
        return "turn_right"
    if diff == 3:
        return "turn_left"
    return "turn_around"


def reason(obs, act):
    return (f"Exit bearing {obs['exit_bearing']} (dist {obs['exit_distance']}); facing {obs['facing']}. "
            f"Ahead={obs['ahead']}, left={obs['left']}, right={obs['right']}. Action: {act}.")


def main():
    episodes_meta, replay = [], {}
    total_reward = won = 0
    n = 12
    for i in range(n):
        env = RaycastDungeonEnv()
        obs = env.reset(seed=200 + i, size=4)
        rng = random.Random(900 + i)
        ep_id = f"ep-{i:03d}"
        turns = []
        ep_reward = 0.0
        step = 0
        while True:
            step += 1
            act = policy(obs, rng, env)
            rs = env.step(act)
            ep_reward += rs.reward
            turns.append({
                "step": step,
                "timestamp": f"2026-05-29T11:{i:02d}:{step:02d}+00:00",
                "observation": obs,
                "reasoning": reason(obs, act),
                "model": MODEL,
                "action": act,
                "reward": round(rs.reward, 4),
                "terminated": rs.terminated,
                "truncated": rs.truncated,
                "info": dict(rs.info),
                "episode_end": ({"total_reward": round(ep_reward, 4), "steps": step,
                                 "status": "completed", "terminal_info": dict(rs.info)}
                                if (rs.terminated or rs.truncated) else None),
            })
            obs = rs.observation
            if rs.terminated or rs.truncated:
                break
        replay[ep_id] = turns
        w = turns[-1]["info"]["won_numeric"] == "1"
        won += w
        total_reward += ep_reward
        episodes_meta.append({"id": ep_id, "seed": 200 + i, "status": "completed",
                              "total_reward": round(ep_reward, 4), "steps": step})

    export = {
        "schema_version": "1", "exported_at": "2026-05-29T11:30:00+00:00",
        "visibility": "gallery_public", "domain_id": "raycast-dungeon",
        "domain_name": "Raycast Dungeon — can an LLM escape a 3D maze?",
        "binding_vow_version": "1.0.0",
        "run": {"id": "sample-run",
                "config": {"domain_id": "raycast-dungeon", "binding_vow_version": "1.0.0",
                           "agent_config": {"model": MODEL}, "num_episodes": n},
                "status": "completed",
                "scores": {"success_rate": round(won / n, 4),
                           "mean_episode_reward": round(total_reward / n, 4)}},
        "episodes": episodes_meta, "traces": {}, "replay": replay,
    }
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(export, open(os.path.join(HERE, "data", "replay.json"), "w"), indent=2)
    with open(os.path.join(HERE, "data", "replay.js"), "w") as f:
        f.write("window.REPLAY = "); json.dump(export, f); f.write(";\n")
    print(f"wrote sample: {n} episodes, {won} escaped ({won/n:.0%}), mean_reward={total_reward/n:.2f}")


if __name__ == "__main__":
    main()
