# train.py
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from curve_env import CurveDrawEnv
from viz import rollout_and_save_gif

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Optional

DEFAULT_GOAL_EPS = 0.02


class MetricsCallback(BaseCallback):
    """
    Collects simple training metrics without writing any files.
    Tracks per-episode returns and success/collision rates across vector envs.
    """

    def __init__(self, n_envs: int):
        super().__init__()
        self.n_envs = n_envs
        self._ep_ret = np.zeros(n_envs, dtype=np.float64)
        self._ep_len = np.zeros(n_envs, dtype=np.int32)
        self._ep_last_dist = np.zeros(n_envs, dtype=np.float64)

        self.ret_history: List[float] = []
        self.len_history: List[int] = []
        self.success_history: List[int] = []
        self.collision_history: List[int] = []
        self.dist_history: List[float] = []

        self.ep_count = 0
        self.window = 100  # moving window for rates

    def _on_step(self) -> bool:
        # These are provided by SB3 during rollout
        # rewards: np.ndarray shape (n_envs,)
        # dones:   np.ndarray shape (n_envs,)
        # infos:   list[dict] of length n_envs
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [{} for _ in range(self.n_envs)])

        if rewards is None or dones is None:
            return True

        self._ep_ret += rewards
        self._ep_len += 1

        # stash the last seen distance-to-goal per env (if provided)
        for i, info in enumerate(infos):
            if isinstance(info, dict) and "distance_to_goal" in info:
                self._ep_last_dist[i] = float(info["distance_to_goal"])

        # when an env is done, log its episode stats
        for i, done in enumerate(dones):
            if done:
                self.ret_history.append(float(self._ep_ret[i]))
                self.len_history.append(int(self._ep_len[i]))
                self.dist_history.append(float(self._ep_last_dist[i]))

                # success/collision flags from final step's info
                info = infos[i] if i < len(infos) else {}
                self.success_history.append(
                    1 if info.get("success", False) else 0)
                self.collision_history.append(
                    1 if info.get("collided", False) else 0)

                # reset counters for that env
                self._ep_ret[i] = 0.0
                self._ep_len[i] = 0
                self._ep_last_dist[i] = 0.0
                self.ep_count += 1

        return True

    # helpers to compute moving-average curves
    @staticmethod
    def _moving_avg(arr: List[float], k: int) -> np.ndarray:
        if len(arr) == 0:
            return np.array([])
        k = max(1, min(k, len(arr)))
        c = np.cumsum(np.insert(np.asarray(arr, dtype=np.float64), 0, 0.0))
        return (c[k:] - c[:-k]) / k

    def plot(self):
        # 1) Episode return (moving average)
        if self.ret_history:
            plt.figure()
            ma = self._moving_avg(self.ret_history, self.window)
            x = np.arange(len(ma)) + 1
            plt.plot(x, ma)
            plt.title(f"Episode Return (moving avg, window={self.window})")
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.tight_layout()
            plt.savefig("result/ret_history.png")

        # 2) Success rate (moving average)
        if self.success_history:
            plt.figure()
            ma = self._moving_avg(self.success_history, self.window)
            x = np.arange(len(ma)) + 1
            plt.plot(x, ma)
            plt.title(f"Success Rate (moving avg, window={self.window})")
            plt.xlabel("Episode")
            plt.ylabel("Rate (0–1)")
            plt.tight_layout()
            plt.savefig("result/success_history.png")

        # 3) Collision rate (moving average)
        if self.collision_history:
            plt.figure()
            ma = self._moving_avg(self.collision_history, self.window)
            x = np.arange(len(ma)) + 1
            plt.plot(x, ma)
            plt.title(f"Collision Rate (moving avg, window={self.window})")
            plt.xlabel("Episode")
            plt.ylabel("Rate (0–1)")
            plt.tight_layout()
            plt.savefig("result/collision_history.png")

        # 4) Final distance-to-goal per episode (moving avg)
        if self.dist_history:
            plt.figure()
            ma = self._moving_avg(self.dist_history, self.window)
            x = np.arange(len(ma)) + 1
            plt.plot(x, ma)
            plt.title(
                f"Final Distance-to-Goal (moving avg, window={self.window})")
            plt.xlabel("Episode")
            plt.ylabel("Distance")
            plt.tight_layout()
            plt.savefig("result/dist_history.png")

    # --- helpers consumed by the curriculum controller -----------------
    def success_rate_since(self, start_idx: int, window: int) -> Optional[float]:
        """Return the moving-average success rate since *start_idx*.

        The curriculum controller resets *start_idx* whenever it promotes the
        agent to a harder stage, so this value only reflects performance on the
        current stage.  Returns ``None`` until we have seen at least ``window``
        episodes for that stage.
        """

        if start_idx < 0:
            start_idx = max(0, len(self.success_history) + start_idx)

        if start_idx >= len(self.success_history):
            return None

        stage_hist = self.success_history[start_idx:]
        if len(stage_hist) < window:
            return None

        ma = self._moving_avg(stage_hist, window)
        if ma.size == 0:
            return None
        return float(ma[-1])

    def episodes_since(self, start_idx: int) -> int:
        """How many episodes have finished since *start_idx*."""

        if start_idx < 0:
            start_idx = max(0, len(self.success_history) + start_idx)
        return max(0, len(self.success_history) - start_idx)


class CurriculumCallback(BaseCallback):
    """
    Амжилтын хөдөлгөөнт дундаж (moving average) >= threshold болсон үед
    саадын шат (n_obstacles range) ахиулдаг callback.
    """

    def __init__(self, stages, metrics_cb: MetricsCallback,
                 window_eps: int = 200, threshold: float = 0.70,
                 check_every_steps: int = 1000,
                 min_episodes_per_stage: int = 0):
        super().__init__()
        self.stages = stages              # [(0,1), (1,2), (2,4), (3,6), ...]
        self.metrics_cb = metrics_cb
        self.window_eps = window_eps
        self.threshold = threshold
        self.check_every_steps = check_every_steps
        self._last_check = 0
        self.stage_idx = 0
        self.min_episodes_per_stage = min_episodes_per_stage
        self._stage_start_idx = 0

    def _on_training_start(self) -> None:
        # эхний шатыг вектортой бүх env-д тохируулна
        self.model.get_env().env_method(
            "set_obstacle_range", self.stages[self.stage_idx])
        print(f"[Curriculum] Start at stage {self.stage_idx+1}/{len(self.stages)} "
              f"n_obstacles={self.stages[self.stage_idx]}")
        self._stage_start_idx = len(self.metrics_cb.success_history)

    def _on_step(self) -> bool:
        # хэт ойр ойрхон шалгахгүй
        if (self.num_timesteps - self._last_check) < self.check_every_steps:
            return True
        self._last_check = self.num_timesteps

        # хангалттай эпизод цугларсан уу?
        episodes_in_stage = self.metrics_cb.episodes_since(self._stage_start_idx)
        min_eps_needed = max(self.window_eps, self.min_episodes_per_stage)
        if episodes_in_stage < min_eps_needed:
            return True

        # сүүлийн window_eps дээрх амжилтын дундаж ( зөвхөн одоогийн шат )
        cur = self.metrics_cb.success_rate_since(
            self._stage_start_idx, self.window_eps)
        if cur is None:
            return True

        stage_no = self.stage_idx + 1
        print(f"[Curriculum] Stage {stage_no}/{len(self.stages)} | "
              f"success_rate={cur:.2f} over last {self.window_eps} episodes "
              f"({episodes_in_stage} episodes seen)")

        if cur >= self.threshold and self.stage_idx < len(self.stages) - 1:
            self.stage_idx += 1
            new_rng = self.stages[self.stage_idx]
            self.model.get_env().env_method("set_obstacle_range", new_rng)
            print(f"[Curriculum] Advance → stage {self.stage_idx+1}/{len(self.stages)} "
                  f"n_obstacles={new_rng} (avg_success={cur:.2f}, "
                  f"episodes_in_stage={episodes_in_stage})")
            self._stage_start_idx = len(self.metrics_cb.success_history)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--save", type=str, default="model/ppo_curve.zip")
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gif", type=str, default="out/eval.gif")
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU policy (no effect if no GPU anyway)")
    args = parser.parse_args()

    # Vectorized env for speed

    def make_env_fn():
        return CurveDrawEnv(with_obstacles=True, goal_eps=DEFAULT_GOAL_EPS)

    vec = make_vec_env(make_env_fn, n_envs=args.n_envs,
                       vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        vec,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        n_steps=1024,
        batch_size=2048,
        learning_rate=1e-4,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=None,
        device="cpu" if args.cpu else "auto",
    )

    metrics_cb = MetricsCallback(n_envs=args.n_envs)

    curriculum_stages = [(0, 1), (1, 2), (2, 4), (3, 6)]

    cur_cb = CurriculumCallback(
        stages=curriculum_stages,
        metrics_cb=metrics_cb,
        window_eps=100,        # 200 биш 100 болго
        threshold=0.60,        # 0.7 биш 0.6 болгож зөөллөх
        check_every_steps=1000, # 2000 биш 1000 алхам тутам шалгахаар хийх
        min_episodes_per_stage=150
    )

    model.learn(total_timesteps=args.timesteps,
                progress_bar=True, callback=[metrics_cb, cur_cb])
    model.save(args.save)

    vec.close()

    # Quick visual eval (single env, deterministic actions)
    env = CurveDrawEnv(with_obstacles=True, goal_eps=DEFAULT_GOAL_EPS,
                       seed=args.seed + 123)
    rollout_and_save_gif(env, model, out_gif=args.gif,
                         max_steps=300, deterministic=True)
    print(f"Saved model to {args.save} and eval to {args.gif}")
    metrics_cb.plot()


if __name__ == "__main__":
    main()
