# torque-bar

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

> ⚠️ In progress — awaiting field testing on a comma 3X.

## What it does

Shows openpilot's steering-torque utilization arc at the bottom of the comma 3X onroad display by default. The arc grows left or right with commanded steering effort and changes from white toward orange near the available steering limit.

## How it works

The comma four UI already includes the complete torque renderer and data flow. This branch reuses that widget in the comma 3X HUD and scales its geometry for the larger 2160×1080 interface; its existing engagement fade and torque-source behavior stay unchanged.

There is no setting or persistent parameter. The arc is part of the standard onroad HUD, just as it is on comma four.

## What changed

- `openpilot/selfdrive/ui/mici/onroad/torque_bar.py` — accepts a display scale while preserving comma four's default rendering.
- `openpilot/selfdrive/ui/onroad/hud_renderer.py` — renders the scaled torque bar above the comma 3X HUD.
