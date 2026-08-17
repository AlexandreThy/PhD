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
| `large_amplitude_reaching.py` | 24 | `LongMove.svg`, `long_move_terminal.svg` |
| `path_constraint.py` | 25 | `PathConstraint.svg`, `PathConstraintCommands.svg` |
| `cost_map_2directions.py` | from `CurrentParts/2Dir.py` | `DLQG_CostMap_90_315.svg` + `.npz` |
| `nonlinearity_index.py` | from `CurrentParts/NonlinearityIndex.py` | `Corr_Plots_{1,2}DLQG.svg` |
| `nonlinearity_ablation.py` | from `CurrentParts/Nonlinearities.ipynb` | `NonlinearityAblation.svg` |
| `centerout_motor_cost.py` | new | `MotorCost_15cm_400ms.svg` + `.npz` |

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
- The starting positions were a cartesian grid of hand positions,
  `x` in `[-10, 10]` and `y` in `[30, 45]`. Over a grid that narrow in `x`, the
  hand height and the distance from the shoulder are collinear, and a cubic fit
  of the cost on either gives the same `R^2` (0.93 for the 90 degree reach), so
  the figure could not show whether the cost follows the starting *posture* or
  just the hand position. The grid is now over the two starting joint angles
  directly — shoulder 10 to 55 deg, elbow 85 to 125 deg — which varies them
  independently. See below.

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
- The peak power index is correlated against the **total** movement cost only.
  The original also regressed it against the motor cost and drew a third figure
  for it; that figure is gone, and `Corr_Plots_2DLQG.svg` is now the total-cost
  scatter that `Corr_Plots_3DLQG.svg` used to hold.

## Path-cost weights and force field strength

Both were chosen by sweeping them and measuring the effect, rather than by eye.

**Straight-path cost** (`WC`, `TAU_PATH` in `common.py`, `WC_SWEEP` in
`path_constraint.py`). `Controllers.FL.compute_path` used to penalise
`[k*(theta_s - theta_s0) - (theta_e - theta_e0)]**2` — straightness in **joint**
space, about one fixed line — and no `(WC, TAU_PATH)` pair could straighten the
hand path with it. A straight joint path maps to a curved hand path: over the
58 cm long movements every fixed joint-space line still bows 5 to 7 cm, and a
grid search over both parameters hit exactly that floor, halving the deviation
at best before it saturated and then grew again.

It now penalises the **cartesian** lateral offset from the start-to-target line,
linearised about a via point that advances along that line on a minimum-jerk
profile. That makes the cost matrix time-varying (`compute_path` returns one
8x8 per timestep) and, being a linear form in the state, it stays an exact LQR
stage cost — `wp * outer(v, v)`, positive semi-definite. At `TAU_PATH = 0.15`
the sweep `WC_SWEEP = (0, 0.003, 0.01, 0.03, 0.1)` now runs from unconstrained
to straight:

| `WC` | 0 | 0.003 | 0.01 | 0.03 | 0.1 |
| --- | --- | --- | --- | --- | --- |
| peak deviation, movement 1 | 13.6 cm | 8.0 | 4.7 | 2.2 | 0.8 |
| peak deviation, movement 2 | 11.9 cm | 4.1 | 1.3 | 0.5 | 0.6 |

The endpoint error is unchanged by the path cost (0.13 to 0.20 cm with motor
noise on, the same as at `WC = 0`). Both movements are saturated past
`WC ~ 0.1`; the second straightens sooner than the first, which is why its last
three sweep curves nearly coincide.

`TAU_PATH` was retuned from 0.2 to 0.15 when `WR_FL` moved to `6e-5`. A cheaper
motor cost makes the same `WC` bite harder, so holding `TAU_PATH` at 0.2 would
have pushed the sweep to straight by the third weight and wasted the last two;
0.15 restores the earlier spacing at the same five weights.

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

In `FFFV.svg` the two axes are cropped to their own data rather than to a shared
range. The field costs several times more than it saves, so the cost with it off
spans about a tenth of the range of the cost with it on; forcing one scale on
both -- which the shared `[0, limit]` equality line used to do -- pushed all
three controllers into the left tenth of the plot. The dashed equal-cost line is
therefore no longer the diagonal.

## Changes made when porting Nonlinearities.ipynb

`nonlinearity_ablation.py` runs the same reach with one or two arm
nonlinearities removed. The notebook carried its own copy of ILQG so it could
switch them; `Controllers/ILQG.py` now takes a `Plant` named tuple instead, and
its default keeps all three, so every other caller is untouched (verified
bit-identical on three deterministic rollouts, one of them under the field).

Switching one off replaces it with its value at the starting posture: the
inertia matrix is frozen at the initial elbow angle so `dM` vanishes, the muscle
force-length and force-velocity gains become 1, and the Coriolis and
centrifugal torques become 0. It applies to the simulated arm as well as to the
model ILQG optimises against, matching the notebook — so the figure asks what
the movement costs when a nonlinearity is *absent*, not what it costs to
mis-model one.

Beyond the weights, two things differ from the notebook:

- It used `WR = 0.5`, 25x the `WR` in `common.py`, and `FFPOW = 4e-4` against
  the current `-3e-4`. Both now come from `common.py` and `force_field.py`.
- Its force-length curve was `exp(+|(l**1.55 - 1) / 0.81|)`, which *rises* as
  the muscle leaves its optimal length — the same stale curve documented above
  for `compute_torque`. The controller integrates `exp(-|...|**2.12)`, which
  falls, so the ablation now scores the dynamics that produced the trajectory.

The notebook's `FFside` argument is not reproduced: the field side is the sign
of `ff_power`, so `--ff-power 3e-4` gives what it called the other side.

## Why the DLQG cost is so much more spread out on the long movements

`long_move_terminal.svg` answers this. The dominant term of the cost is
`WP * (dtheta_s**2 + dtheta_e**2)` with `WP = 20000`, and its sensitivity to
trial-to-trial jitter is its derivative, `2 * WP * dtheta` — proportional to how
far off the *mean* landing point is, not to the size of the noise.

ILQG and FL land on the target (mean terminal error under 0.05 deg), so they sit
at the bottom of that parabola where the derivative vanishes: their target term
is pure `WP * noise**2`, which behaves like a scaled chi-square with mean equal
to its own SD (0.44 +/- 0.40 for ILQG on movement 2). DLQG ends 4.3 deg short at
the elbow, far up the parabola, so the *same* jitter is amplified. On movement 2,
`2 * WP * |mean| * sd` predicts an SD of 19.1 against 20.3 measured, and the term
accounts for 100% of the variance of the total.

So the spread is not noise sensitivity in the control sense — the terminal
jitter of DLQG (0.36 deg) is comparable to that of the other two (about 0.2 deg).
It is the systematic offset multiplying it. Note also that in *relative* terms
DLQG is not the most variable: its CV is 0.16 against 0.44 for ILQG, whose cost
is small and entirely noise-driven.

FL is the reverse case: 81% of its variance is the motor term, which is large
(89.9 of a total of 90.3 on movement 2) but very stable, giving it the lowest CV
of the three at 0.010. That term is large because FL is scored with the shared
`WR` while it optimises with `WR_FL`, 300 times smaller — see the note under the
cost function weights.

## What the 2-direction cost map shows

Every reach in `DLQG_CostMap_90_315.svg` is the same 15 cm in the same
direction, so the amplitude and the direction cannot explain any of the
variation across a panel — only the posture the reach starts from can.

Sampling the joint angles independently separates the two joints, and the
result is that the cost is close to a function of the **elbow angle** alone:
across the 85 to 125 degree elbow range the mean cost changes 14-fold for the
90 degree reach and 8-fold for the 315 degree reach, while the spread over the
whole 10 to 55 degree shoulder range at a fixed elbow angle is a fraction of
that. This is why the maps are banded horizontally.

The two directions band in *opposite* senses: extending the arm makes the 90
degree reach cheaper and the 315 degree reach dearer. A posture change that
helps one direction hurts the other, which is what rules out reading the effect
as a plain "some places in the workspace are expensive".

Both ratios are quoted off the mean profile over shoulder angles, not off the
cheapest and dearest single cells — those are the noisiest estimates on the
grid and put the ratios near 45 and 33 at 6 trials per cell.

## Cost function weights

Set once in `common.py` and used by every script:

- `WR_FL = 1e-4` — motor cost of FL.
- `WR = 0.01` — motor cost of ILQG and DLQG.

These replace values that varied between notebook cells: the FL motor cost
appeared as `1e-8` (sensitivity analysis) and `1e-3` (one cost polar cell), and
the ILQG/DLQG motor cost appeared as `0.5` in two cells. The cost polar cell
that used FL `r=1e-3` was a duplicate of the `1e-4` one and is not repeated
here — `centerout_cost_polar.py` covers that condition once.

Note the asymmetry when *scoring*: FL is optimised with `WR_FL` but scored with
`WR`, like the other two. `Cost_function` defaults to `WR`, so every script that
compares controllers already does this; the motor bars in
`Kinematiccenterout.svg` and the cost scatter in `FFFV.svg` are on that common
yardstick. Scoring each controller under the weights it optimised would compare
three different quantities.

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
