# curve_env.py
from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import scale as shp_scale

Coordinate = Tuple[float, float]


def _rand_point(margin: float = 0.05) -> Coordinate:
    return (random.uniform(margin, 1 - margin), random.uniform(margin, 1 - margin))


def _segment_intersects_polys(p0: Coordinate, p1: Coordinate, polys: List[Polygon]) -> bool:
    seg = LineString([p0, p1])
    for poly in polys:
        if seg.crosses(poly) or seg.within(poly) or seg.intersects(poly):
            # If segment merely touches boundary, consider it a collision too
            if seg.intersection(poly).length > 0 or not seg.intersection(poly).is_empty:
                return True
    return False


def _make_obstacles(n: int,
                    min_r: float = 0.05,
                    max_r: float = 0.15,
                    keep_inside: Tuple[float, float, float,
                                       float] = (0.0, 0.0, 1.0, 1.0)
                    ) -> List[Polygon]:
    """
    Build random convex-ish obstacles as scaled regular polygons.
    """
    xmin, ymin, xmax, ymax = keep_inside
    polys: List[Polygon] = []
    for _ in range(n):
        # center
        cx = random.uniform(xmin + max_r, xmax - max_r)
        cy = random.uniform(ymin + max_r, ymax - max_r)
        # base regular polygon (octagon); then scale by random rx, ry
        base = Point(cx, cy).buffer(1.0, resolution=8)  # unit-ish
        rx = random.uniform(min_r, max_r)
        ry = random.uniform(min_r, max_r)
        poly = shp_scale(base, rx, ry, origin=(cx, cy))
        # clamp inside [0,1]^2 (roughly handled by choosing center/margins)
        polys.append(poly)
    return polys


class CurveDrawEnv(gym.Env):
    """
    State: concatenation of
      - current point (x, y)
      - goal (gx, gy)
      - previous direction (dx, dy) (unit)
      - normalized distance-to-goal (scalar)
      - LiDAR polar distances to obstacles (K beams), normalized to [0,1] with max_range

    Action (Box): [delta_heading, step_scale]
      - delta_heading in radians, clipped to [-max_turn, +max_turn]
      - step_scale in [0, 1] (scaled to [min_step, max_step])

    Episode ends when:
      - reach distance <= goal_eps
      - step hits or crosses any obstacle
      - out of bounds
      - step count limit
    """
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self,
                 with_obstacles: bool = True,
                 n_obstacles: Tuple[int, int] = (2, 6),
                 lidar_beams: int = 32,
                 max_range: float = 0.5,
                 min_step: float = 0.01,
                 max_step: float = 0.05,
                 max_turn: float = math.radians(45),
                 goal_eps: float = 0.07,
                 max_steps: int = 200,
                 seed: Optional[int] = None):
        super().__init__()
        self.with_obstacles = with_obstacles
        self.n_obstacles_rng = n_obstacles
        self.lidar_beams = lidar_beams
        self.max_range = max_range
        self.min_step = min_step
        self.max_step = max_step
        self.max_turn = max_turn
        self.goal_eps = goal_eps
        self.max_steps = max_steps

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(seed)

        # Observation space
        # (x,y), (gx,gy), (dx,dy), dist_norm, K beams
        obs_dim = 2 + 2 + 2 + 1 + lidar_beams
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Action space: delta_heading, step_scale
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([+1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.start: Coordinate = (0.1, 0.1)
        self.goal: Coordinate = (0.9, 0.9)
        self.cur: Coordinate = self.start
        self.prev_dir: np.ndarray = np.array([1.0, 0.0], dtype=np.float32)
        self.polys: List[Polygon] = []
        self.path: List[Coordinate] = []
        self.steps = 0

    # ---- helpers -------------------------------------------------------------

    def _lidar(self) -> np.ndarray:
        """
        Cast K rays around current heading to detect nearest obstacle within max_range.
        We’ll cast relative to prev_dir’s angle to give agent a forward-facing sense.
        """
        K = self.lidar_beams
        origin = Point(self.cur)
        # Forward-facing arc (±90° around heading), or full 360?
        # Use full 360 for simplicity.
        angles = np.linspace(-math.pi, math.pi, K, endpoint=False)
        # Heading angle from prev_dir
        base_theta = math.atan2(self.prev_dir[1], self.prev_dir[0])

        # default: 1.0 (no obstacle within range)
        dists = np.ones(K, dtype=np.float32)
        if not self.polys:
            return dists

        for i, a in enumerate(angles):
            theta = base_theta + a
            # Ray end point
            ex = self.cur[0] + self.max_range * math.cos(theta)
            ey = self.cur[1] + self.max_range * math.sin(theta)
            ray = LineString([self.cur, (ex, ey)])
            hit_dist = self.max_range
            for poly in self.polys:
                inter = ray.intersection(poly.boundary)
                if inter.is_empty:
                    continue
                if inter.geom_type == "Point":
                    d = origin.distance(inter)
                    if d < hit_dist:
                        hit_dist = d
                elif inter.geom_type == "MultiPoint":
                    for p in inter.geoms:
                        d = origin.distance(p)
                        if d < hit_dist:
                            hit_dist = d
            dists[i] = hit_dist / self.max_range
        return dists

    def _obs(self) -> np.ndarray:
        x, y = self.cur
        gx, gy = self.goal
        v = np.array([gx - x, gy - y], dtype=np.float32)
        dist = np.linalg.norm(v) + 1e-9
        dist_norm = np.float32(min(1.0, dist / math.sqrt(2)))
        if dist > 0:
            gdir = v / dist
        else:
            gdir = np.zeros_like(v)
        lidar = self._lidar()
        obs = np.concatenate([
            np.array([x, y], dtype=np.float32) * 2 -
            1,              # map [0,1]->[-1,1]
            np.array([gx, gy], dtype=np.float32) * 2 - 1,
            # already unit
            self.prev_dir.astype(np.float32),
            np.array([dist_norm], dtype=np.float32),
            lidar.astype(np.float32)
        ], axis=0)
        return obs

    # ---- gym API -------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(seed)

        # Random start/goal with separation
        ok = False
        for _ in range(200):
            s = _rand_point(0.07)
            g = _rand_point(0.07)
            if math.dist(s, g) > 0.5:
                ok = True
                break
        if not ok:
            s = (0.1, 0.1)
            g = (0.9, 0.9)

        self.start = s
        self.goal = g
        self.cur = self.start
        self.prev_dir = np.array([1.0, 0.0], dtype=np.float32)

        # Obstacles
        self.polys = []
        if self.with_obstacles:
            nobs = random.randint(
                self.n_obstacles_rng[0], self.n_obstacles_rng[1])
            # Try a few times to avoid instantly blocking start/goal
            for _ in range(50):
                candidate = _make_obstacles(nobs)
                if Point(self.start).within(Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])) and \
                   Point(self.goal).within(Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])):
                    # Ensure start/goal not inside obstacles
                    if not any(Point(self.start).within(p) for p in candidate) and \
                       not any(Point(self.goal).within(p) for p in candidate):
                        self.polys = candidate
                        break

        self.path = [self.start]
        self.steps = 0
        obs = self._obs()
        info = {"path": self.path.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Action = [turn_scalar in [-1,1], step_scalar in [0,1]]
        """
        import math
        import numpy as np

        # --- Hyperparams ----------------------------------------------------------
        goal_eps = getattr(self, "goal_eps", 0.03)
        goal_bonus = getattr(self, "goal_bonus",30.0)
        progress_scale = getattr(self, "progress_scale", 6.0)
        small_step_penalty = getattr(self, "small_step_penalty", 0.0)

        # --- Parse action ---------------------------------------------------------
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 2:
            # хамгаалалт: буруу хэмжээтэй ирвэл богино алхам, 0 эргэлт гэж үзнэ
            a = np.array([0.0, 0.0], dtype=np.float32)

        turn_scalar = float(np.clip(a[0], -1.0, 1.0))
        step_scalar = float(np.clip(a[1],  0.0, 1.0))

        dtheta = turn_scalar * self.max_turn
        step_len = self.min_step + step_scalar * \
            (self.max_step - self.min_step)

        # --- Propose new state ----------------------------------------------------
        theta0 = math.atan2(self.prev_dir[1], self.prev_dir[0])
        theta = theta0 + dtheta
        new_dir = np.array(
            [math.cos(theta), math.sin(theta)], dtype=np.float32)

        new_pt = (
            float(self.cur[0] + step_len * new_dir[0]),
            float(self.cur[1] + step_len * new_dir[1]),
        )

        # --- Checks ---------------------------------------------------------------
        out_of_bounds = not (
            0.0 <= new_pt[0] <= 1.0 and 0.0 <= new_pt[1] <= 1.0)

        collided = False
        if (not out_of_bounds) and self.with_obstacles and len(self.polys) > 0:
            seg = LineString([self.cur, new_pt])
            for poly in self.polys:
                # boundary-г шүргэхийг ч мөргөлдөөнд тооцно
                if seg.crosses(poly) or seg.within(poly) or seg.intersects(poly):
                    inter = seg.intersection(poly)
                    if (not inter.is_empty) or getattr(inter, "length", 0.0) > 0.0:
                        collided = True
                        break

        # --- Reward ---------------------------------------------------------------
        r = 0.0
        old_dist = math.dist(self.cur, self.goal)
        new_dist = math.dist(new_pt, self.goal)

        progress = max(0.0, old_dist - new_dist)
        r += progress_scale * progress

        # smooth curvature bonus / penalty
        turn_ratio = abs(dtheta) / max(1e-6, self.max_turn)
        mu, sigma = 0.35, 0.25
        curvature_bonus = math.exp(-0.5 *
                                   ((turn_ratio - mu) / max(1e-6, sigma)) ** 2)
        r += 0.5 * curvature_bonus
        if turn_ratio > 0.7:
            r -= 0.6 * (turn_ratio - 0.7) / 0.3

        r += small_step_penalty

        terminated = False
        truncated = False
        success = False

        # --- Transitions / Termination -------------------------------------------
        if out_of_bounds:
            r -= 2.0
            truncated = True
        elif collided:
            r -= 10.0
            terminated = True
        else:
            # commit move
            self.cur = new_pt
            self.path.append(self.cur)
            self.prev_dir = new_dir

            # goal reached
            if new_dist <= goal_eps:
                r += goal_bonus
                success = True
                terminated = True

        # max steps
        self.steps += 1
        if self.steps >= self.max_steps and not (terminated or truncated):
            if new_dist > goal_eps:
                r -= 0.5
            truncated = True

        obs = self._obs()
        info = {
            "distance_to_goal": new_dist,
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "success": success,
            "path": self.path.copy(),
        }
        return obs, float(r), terminated, truncated, info

    def set_obstacle_range(self, rng: Tuple[int, int]):
        """
        Curriculum-д хэрэгтэй: саадын тооны (min,max) интервалыг шууд солих.
        Жич: өөрчлөлт нь дараагийн reset() дээр үйлчилнэ.
        """
        assert isinstance(rng, tuple) and len(rng) == 2
        self.n_obstacles_rng = rng

    def render(self):
        pass
