# spinning-steering-wheel

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

> ⚠️ In progress — awaiting field testing on a comma 3X.

## What it does

Rotates the comma 3X onroad display's existing top-right steering-wheel icon with the car's measured steering angle.

The Chill/Experimental mode button otherwise behaves as before. There is no setting or persistent parameter.

## How it works

The mode button draws its current icon around a centered origin and rotates it directly from `carState.steeringAngleDeg`, matching the physical wheel direction.

## What changed

- `openpilot/selfdrive/ui/onroad/exp_button.py` — rotates the existing mode-button icon around its center.
