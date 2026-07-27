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

## Cost function weights

Set once in `common.py` and used by every script:

- `WR_FL = 1e-4` — motor cost of FL.
- `WR = 0.1` — motor cost of ILQG and DLQG.

These replace values that varied between notebook cells: the FL motor cost
appeared as `1e-8` (sensitivity analysis) and `1e-3` (one cost polar cell), and
the ILQG/DLQG motor cost appeared as `0.5` in two cells. The cost polar cell
that used FL `r=1e-3` was a duplicate of the `1e-4` one and is not repeated
here — `centerout_cost_polar.py` covers that condition once.

## Known issue: ILQG diverges on `longmovement_2`

`large_amplitude_reaching.py` and `path_constraint.py` raise
`ControllerDiverged` for ILQG on `longmovement_2` (start `[-22.6, 45.5]`,
target `[30.5, 21.1]`).

This is a property of the controller, not of these scripts. On ILQG's first
improvement step the command reaches `|u| ~ 1.87`, which drives the joint
trajectory to a posture where the normalised muscle length `l` becomes slightly
negative (about `-0.017`). `l**1.55` of a negative number is `NaN`, the state
becomes non-finite, and the `np.linalg.eig` call in `step3` raises. FL and DLQG
complete the same movement without trouble, and both controllers plus ILQG
complete `longmovement_1`.

Until the force-length curve handles `l < 0`, run these two scripts on the
movement that converges:

```bash
python Final_Code/large_amplitude_reaching.py --movements 1
python Final_Code/path_constraint.py --movements 1
```

Fixing it means deciding what the muscle model should do when a muscle is driven
past zero length (clamp `l`, or reject the step), which changes controller
behaviour and so was left alone here.

## Controller performance

`Controllers/FL.py`, `Controllers/ILQG.py` and `Controllers/LQGControllers.py`
were optimised without changing their behaviour: for the same inputs they return
**bit-identical** outputs to before, verified on all four returned arrays of each
controller, with and without the force field.

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
