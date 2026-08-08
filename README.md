# spinning-steering-wheel

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

> ⚠️ In progress — awaiting field testing on a comma 3X.

## What it does

Adds the comma four's animated steering-wheel indicator to the bottom-left of the comma 3X onroad display. It rotates with the car's measured steering angle, fades and slides into view while openpilot is engaged, and hides when disengaged.

The existing top-right Chill/Experimental mode button is unchanged. There is no setting or persistent parameter.

## How it works

The comma 3X HUD loads the existing comma four wheel asset at its native comma 3X UI scale. Its rotation comes directly from `carState.steeringAngleDeg`, using the same direction and engagement animation as the comma four renderer.

## What changed

- `openpilot/selfdrive/ui/onroad/hud_renderer.py` — renders and animates the rotating steering-wheel indicator.
