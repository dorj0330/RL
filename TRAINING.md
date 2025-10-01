# Training the Curve Drawing Agent

The PPO training script (`train.py`) already contains a four-stage obstacle
curriculum.  Each stage increases the number of obstacles that can spawn inside
the unit square:

1. **Stage 1** – `(0, 1)` obstacles (free space or one mild obstacle)
2. **Stage 2** – `(1, 2)` obstacles
3. **Stage 3** – `(2, 4)` obstacles
4. **Stage 4** – `(3, 6)` obstacles

The curriculum controller promotes the agent to the next stage only after it has
completed **at least 150 episodes** on the current stage and the moving-average
success rate over the most recent 100 episodes exceeds the success threshold
(default `0.60`).  This guards against the agent skipping ahead before it has
mastered the current difficulty.

## Quick start

```bash
# Train with 16 parallel environments for 300k environment steps
python train.py --timesteps 300000 --n_envs 16
```

During training you will see log lines such as:

```
[Curriculum] Stage 1/4 | success_rate=0.64 over last 100 episodes (152 episodes seen)
[Curriculum] Advance → stage 2/4 n_obstacles=(1, 2) (avg_success=0.64, episodes_in_stage=152)
```

These messages report the rolling success rate for the current stage and note
when the agent graduates to the next stage.  Once stage 4 is reached the
controller continues to print the measured success rate so you can monitor
progress in the hardest setting.

## Tips for reaching the goal marker consistently

* **Run for enough steps.** The defaults (`300k` steps with 16 environments)
  usually progress through all stages, but you can extend the run if stage 4
  success remains below the threshold:
  
  ```bash
  python train.py --timesteps 600000
  ```

* **Resume from a checkpoint.** If you already have a partial model, load it in
  `train.py` (see Stable-Baselines3's `PPO.load` + `set_parameters`).  The
  curriculum will keep its progress and continue counting episodes from the
  current stage.

* **Inspect metrics.** After training finishes the script stores learning curves
  under `result/` so you can verify collision, success, and distance trends.

* **Reward shaping.** `CurveDrawEnv` now includes a small per-step penalty and a
  failure penalty that scales with how far the final point is from the goal. This
  makes it costly for the policy to stop short.  The evaluation harness prints
  the final `shortfall` distance so you can confirm convergence to the goal.

* **Evaluate visually.** Use `test_eval.py` with the saved model to render the
  provided static scenarios:

  ```bash
  python test_eval.py
  ```

  PNGs are written to the `out/` directory and should now show the path touching
  the goal markers thanks to the stricter success tolerance.
