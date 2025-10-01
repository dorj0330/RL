# run_static_tests.py
"""
Run a few static test scenarios with a saved PPO model (ppo_curve.zip),
and save both a GIF (full rollout) and a PNG (final frame) for each case
into ./out/.

Requirements:
- ppo_curve.zip (trained with train.py from earlier message)
- curve_env.py and viz.py in the same folder

Usage:
  python run_static_tests.py
"""

import os
from typing import List, Tuple, Optional
import imageio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

from stable_baselines3 import PPO

from curve_env import CurveDrawEnv, Coordinate
from viz import rollout_and_save_gif  # uses your existing drawer


# ---------- Small helper to draw a final PNG ----------
def _draw_env(ax, env: CurveDrawEnv, path: List[Coordinate]):
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Curve Drawing RL — Final State")

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


# ---------- Fixed env wrapper (keeps your original env intact) ----------
class FixedCurveEnv(CurveDrawEnv):
    def __init__(
        self,
        start: Coordinate,
        goal: Coordinate,
        polys: List[Polygon],
        **kwargs
    ):
        super().__init__(with_obstacles=True, **kwargs)
        self._fixed_start = start
        self._fixed_goal = goal
        self._fixed_polys = polys

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        # Override with fixed values
        self.start = tuple(self._fixed_start)  # type: ignore
        self.cur = self.start
        self.goal = tuple(self._fixed_goal)    # type: ignore
        self.polys = self._fixed_polys
        self.path = [self.start]
        # Rebuild obs
        obs = self._obs()
        info = {"path": self.path.copy()}
        return obs, info


def _circle(cx: float, cy: float, r: float) -> Polygon:
    return Point(cx, cy).buffer(r, resolution=48)


def run_one_case(
    model_path: str,
    out_dir: str,
    case_name: str,
    start: Coordinate,
    goal: Coordinate,
    obstacles: List[Polygon],
    max_steps: int = 300,
    deterministic: bool = True,
    seed: Optional[int] = 1234,
):
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, f"{case_name}.gif")
    png_path = os.path.join(out_dir, f"{case_name}.png")

    # Build fixed env
    env = FixedCurveEnv(start=start, goal=goal,
                        polys=obstacles, max_steps=max_steps, seed=seed)
    model = PPO.load(model_path, device="auto")

    # 1) Save GIF of the rollout
    # rollout_and_save_gif(env, model, out_gif=gif_path,
    #                      max_steps=max_steps, deterministic=deterministic)
    # print(f"[{case_name}] Saved GIF -> {gif_path}")

    # 2) Re-run quickly to capture the final frame into PNG
    obs, info = env.reset(seed=seed)
    path = info.get("path", [env.start])
    steps = 0
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(action)
        path = info.get("path", path)
        steps += 1
        if terminated or truncated:
            break

    # Draw final PNG
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    _draw_env(ax, env, path)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"[{case_name}] Saved PNG -> {png_path}")


def main():
    MODEL = "model/ppo_curve.zip"
    OUT_DIR = "out"

    # ---- Define STATIC scenarios here ----
    scenarios = [
        # Simple: one centered circle, diagonal path
        dict(
            name="case1_diagonal_circle",
            start=(0.1, 0.1),
            goal=(0.9, 0.9),
            obstacles=[],
        ),
        # Two obstacles offset
        dict(
            name="case2_two_circles",
            start=(0.1, 0.2),
            goal=(0.9, 0.8),
            obstacles=[_circle(0.4, 0.6, 0.10), _circle(0.7, 0.45, 0.08)],
        ),
        # Narrow channel
        dict(
            name="case3_channel",
            start=(0.1, 0.5),
            goal=(0.9, 0.5),
            obstacles=[
                _circle(0.5, 0.35, 0.12),
                _circle(0.5, 0.65, 0.12),
            ],
        ),
        # Curvy detour: cluster near center
        dict(
            name="case4_cluster",
            start=(0.15, 0.15),
            goal=(0.85, 0.85),
            obstacles=[
                _circle(0.45, 0.45, 0.08),
                _circle(0.55, 0.52, 0.07),
                _circle(0.52, 0.40, 0.06),
            ],
        ),
    ]

    for sc in scenarios:
        run_one_case(
            model_path=MODEL,
            out_dir=OUT_DIR,
            case_name=sc["name"],
            start=sc["start"],
            goal=sc["goal"],
            obstacles=sc["obstacles"],
            max_steps=300,
            deterministic=True,
            seed=2025,  # fixed seed for reproducibility
        )

    print("\nAll static cases done. Check the 'out/' folder.")


if __name__ == "__main__":
    main()
