# better-green-lights

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

Makes **experimental-mode green-light launches start sooner and pull harder**. Stock end-to-end mode waits for the model's `shouldStop` bit to release and then eases away on the model's lazy back-loaded plan; this branch reads the model's *intent* directly and launches on it, cutting the "light's green and we're still sitting here" pause.

*(personal idea)*

## How it works

Two mechanisms, both in the longitudinal planner and both active only in experimental mode:

- **Green-light anticipation** — at a red light the model's planned path is a short stub (2–5 m); when the light changes, the path *explodes* to 30–60 m about 1.5–2 s **before** the laggy `shouldStop` bit releases (measured in field data, routes 37/38). A filtered detector on the path-end length (>20 m open, ~0.5 s of sustained confidence) clears `shouldStop` early, letting the car creep off; if the path re-collapses below 10 m — the model changed its mind — the hold re-engages at creep speed.
- **Launch assist** — once moving, the model's speed plan still front-loads hesitation: it plans several near-zero seconds before committing. If the plan says we'll genuinely be moving (>2 m/s at the 3.5 s mark), the plan is *time-shifted* to skip the dead time at its head and the acceleration is recomputed from the shifted plan, capped at 1.5 m/s² and faded out entirely by 2 m/s so it only shapes the first car length of the launch.

## What changed

- `openpilot/selfdrive/controls/lib/longitudinal_planner.py` — the only file: `LAUNCH_*` constants, the filtered path-length open/close detector with its anticipation latch, and the time-shifted launch-assist accel floor applied to the e2e accel target.
