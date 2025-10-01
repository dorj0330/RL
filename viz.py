# viz.py
from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np
import imageio
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from curve_env import CurveDrawEnv, Coordinate


def _draw_env(ax, env: CurveDrawEnv, path: List[Coordinate]):
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Curve Drawing RL")

    # Obstacles
    for p in env.polys:
        if isinstance(p, Polygon):
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, alpha=0.6)

    # Start / Goal
    sx, sy = env.start
    gx, gy = env.goal
    ax.scatter([sx], [sy], s=60, marker="o")
    ax.scatter([gx], [gy], s=60, marker="*", zorder=5)

    # Path
    if len(path) >= 2:
        xs = [q[0] for q in path]
        ys = [q[1] for q in path]
        ax.plot(xs, ys, linewidth=2)

    # Current
    cx, cy = env.cur
    ax.scatter([cx], [cy], s=30, marker="o", zorder=6)


def rollout_and_save_gif(env: CurveDrawEnv, model, out_gif: str, max_steps: int = 300, deterministic: bool = True):
    frames = []
    obs, info = env.reset()
    path = info.get("path", [env.start])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    for _ in range(max_steps):
        _draw_env(ax, env, path)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        path = info.get("path", path)

        if terminated or truncated:
            _draw_env(ax, env, path)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            break

    imageio.mimsave(out_gif, frames, fps=30)
    plt.close(fig)
