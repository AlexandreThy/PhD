# Final_Code

The figures from `CurrentParts/PaperPlot.ipynb`, split into one script per task.
Every script shares its parameters, cost function and plotting style through
`common.py`, so a weight is defined in exactly one place.

## Scripts

| Script | Notebook cells | Output |
| --- | --- | --- |
| `centerout_trajectories.py` | 2, 3, 4 | `{ILQG,FL,DLQG}_Centerout.svg` and `*_Centerout_mean.svg` |
| `centerout_cost_polar.py` | 6, 7, 8, 11, 12, 14, 15 | `Cfy40_<amp>cm_<dur>ms.svg` and `*_cost.npz` |
| `velocity_profiles.py` | 17 | `Kinematiccenterout.svg` |
| `force_field.py` | 20 | `FF3Controllers.svg`, `FFFV.svg` |
| `sensitivity_analysis.py` | 22 | `SensitivityAnalysis.svg` |
| `large_amplitude_reaching.py` | 24 | `LongMove.svg` |
| `path_constraint.py` | 25 | `PathConstraint.svg`, `PathConstraintCommands.svg` |
| `cost_map_2directions.py` | from `CurrentParts/2Dir.py` | `DLQG_CostMap_90_315.svg` + `.npz` |
| `nonlinearity_index.py` | from `CurrentParts/NonlinearityIndex.py` | `Corr_Plots_{1,2,3}DLQG.svg` |

`nonlinearity_index.py` correlates against the cost written by
`centerout_cost_polar.py`, so run that for the 15 cm / 400 ms condition first.

Figures are written to `Final_Code/figures/`.

## Usage

```bash
python Final_Code/centerout_cost_polar.py                        # all conditions
python Final_Code/centerout_cost_polar.py --amplitude 15 --duration 0.4
python Final_Code/centerout_cost_polar.py --num-sim 5 --jobs 1    # quick check
```

Common flags:

- `--num-sim N` repetitions per condition (the notebook used 100).
- `--jobs N` worker processes; defaults to `cpu_count - 1`, `--jobs 1` runs
  serially and is the one to use when debugging, because a traceback from a
  worker process is harder to read.
- `--outdir DIR` where to write the figures.
- `--no-show` save without opening a window.

## Changes made when porting 2Dir.py and NonlinearityIndex.py

Both now take their weights from `common.py` like every other script. Beyond
the weights, the following had to change.

`2Dir.py` did not run as written:

- `DLQG_6Muscles(..., r=1e-8, ...)` raised `TypeError`, since the parameter is
  named `r1`. It now passes `r1 = WR`.
- `Cost_function(x, u, ...)` referenced `x` and `u`, which are not defined in
  that scope — the call returns `xLQG, yLQG, uDLQG, z`. It now scores the DLQG
  state and command.
- Every title said "iLQG" although the script calls `DLQG_6Muscles`, so the
  figure and its filename now say DLQG.

`NonlinearityIndex.py`:

- `compute_torque` inlined `fl = exp(+|(l**1.55 - 1) / 0.81|)`, an older
  force-length curve that *rises* as the muscle leaves its optimal length. The
  controllers integrate `exp(-|...|**2.12)`, which falls, so the torque did not
  match the dynamics that produced the trajectory. It now calls
  `muscle_force_scaling` from `Controllers/ILQG.py`. This changes the numbers;
  revert that one call to restore the old behaviour.
- The third figure plotted the DLQG fit on the ILQG panel.
- The correlations read `Costdata.npz` and `Costr.npz` from the repository root,
  which hold results from the old weights. They now read
  `figures/Cfy40_15cm_400ms_cost.npz`, the matching condition at the current
  weights, and say so if it is missing.

## Via-path weights and force field strength

Both were chosen by sweeping them and measuring the effect, rather than by eye.

**Via-path cost** (`WC`, `TAU_PATH` in `common.py`, `WC_SWEEP` in
`path_constraint.py`). The old settings (`TAU_PATH = 0.03`, `WC <= 0.01`) moved
the peak lateral deviation of the long movements by 0.04 percent, which is not
visible on a plot: the cost decayed away within about 100 ms of a 600 ms
movement, and the weights were some two orders of magnitude too small. With
`TAU_PATH = 0.6`, so that the constraint acts throughout the movement, the
sweep `WC_SWEEP = (0, 0.1, 0.3, 1, 3)` straightens the hand path step by step —
peak deviation falls by roughly 0, 11, 25, 44 and 55 percent — while the
endpoint error stays under 0.03 cm.

Past `WC ~ 3` the deviation stops improving and then grows again, so the sweep
stops there. There is also a floor near 6 cm that no weight can beat:
`compute_path` penalises `[k*(theta_s - theta_s0) - (theta_e - theta_e0)]**2`,
which is straightness in **joint** space, and a straight joint path maps to a
curved hand path. Straightening the hand path further would need the cost
expressed in cartesian coordinates, which is a different method.

**Force field strength** (`FF_POWER` in `force_field.py`). Picked so the
controllers rank

- `FL < ILQG < DLQG` by cost with the field **on**, and
- `ILQG < FL < DLQG` with it **off**.

The window that gives both is narrow. As the field strengthens, ILQG's cost
grows fastest and DLQG's slowest, so ILQG overtakes FL near `6.5e-4` and
overtakes DLQG near `9e-4`; only between those does ILQG sit second in both
conditions. The old `4e-4` was below the window (ILQG was still cheapest with
the field on) and `2e-3` was above it (DLQG became the *cheapest*, and the
lateral excursion reached 15 cm).

`FF_POWER = 7.5e-4` sits in the middle of the window with both gaps well
resolved over 150 trials — ILQG minus FL is `+13.4 +/- 1.5`, DLQG minus ILQG is
`+23.6 +/- 1.4` — while the lateral excursion stays near 10 cm and the terminal
error near 2 cm. `7e-4` is the alternative if a smaller perturbation matters
more than the size of the ILQG-FL gap; below `6.5e-4` that gap falls into the
noise.

Throughout the window ILQG's lateral excursion is slightly *smaller* than FL's
yet it costs more, so what ILQG loses under the field is terminal accuracy
rather than path deviation.

## Cost function weights

Set once in `common.py` and used by every script:

- `WR_FL = 1e-4` — motor cost of FL.
- `WR = 0.01` — motor cost of ILQG and DLQG.

These replace values that varied between notebook cells: the FL motor cost
appeared as `1e-8` (sensitivity analysis) and `1e-3` (one cost polar cell), and
the ILQG/DLQG motor cost appeared as `0.5` in two cells. The cost polar cell
that used FL `r=1e-3` was a duplicate of the `1e-4` one and is not repeated
here — `centerout_cost_polar.py` covers that condition once.

## ILQG on `longmovement_2`: line search

ILQG used to crash on `longmovement_2` (start `[-22.6, 45.5]`, target
`[30.5, 21.1]`). Its Newton step is undamped, so on that movement the very first
improvement reached `|u| ~ 1.87` and drove the arm to a posture where the
normalised muscle length `l` went slightly negative (about `-0.017`). `l**1.55`
of a negative number is `NaN`, so the state became non-finite and the
`np.linalg.eig` call in `step3` raised.

`simulate_ILQG` now backtracks: if the full step rolls out to a non-finite
trajectory, the step is halved until it does not, which is the usual safeguard
in iLQG. A step that is already finite is taken whole.

The intervention is small and targeted:

- `longmovement_2` needs **one** halving, on iteration 0 only, then converges
  normally at iteration 62 with `|u| ~ 0.72` and an endpoint error of `0.0000` cm.
- The center-out movements need **zero** halvings, so their iterates are
  unchanged — the controllers still return bit-identical output to before
  (verified, see below).

The muscle model itself was not touched, so nothing changes about what the model
says when a muscle is driven past zero length; the optimiser simply no longer
steps there. `MAX_BACKTRACKS` in `Controllers/ILQG.py` caps the halving, and a
step that cannot be made finite raises with the iteration number.

Both scripts accept `--movements 1` / `--movements 2` to run a single movement.

## Controller performance

`Controllers/FL.py`, `Controllers/ILQG.py` and `Controllers/LQGControllers.py`
were optimised without changing their behaviour: for the same inputs they return
**bit-identical** outputs to before, verified on all four returned arrays of each
controller, with and without the force field. The ILQG line search above is the
one deliberate behaviour change, and it only engages where the controller
previously crashed.

The speedups came from removing work whose result was unused or constant:

- Muscle model constants (moment arms, `l0`, `theta0`) were rebuilt from nested
  Python lists on every call of the dynamics; they are now module constants.
- `FL.nonlinear_transform_command` called `np.linalg.pinv` — an SVD — on a
  constant matrix once per timestep.
- `FL.compute_next_state` computed an acceleration through `np.linalg.solve`
  that was overwritten before use.
- `ILQG.step5` called `f()` once per timestep to fill a variable that was never
  read.
- The estimator's delayed-state matrices and the noise covariances were
  reallocated every timestep although they only depend on `dt` and the delay.

Roughly 5x for FL, 2.5x for ILQG and 2.2x for DLQG when measured on an idle
machine; treat the exact numbers as indicative, since timings on this machine
varied by more than 2x with background load.

The scripts add a second, larger speedup: repetitions are independent, so they
are distributed over processes (`--jobs`).
