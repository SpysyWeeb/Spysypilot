# AOL — Always-On-Lateral

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

**Decouples steering from cruise control.** With AOL, openpilot's lane keeping can be active while longitudinal control is not engaged — press cruise MAIN and the car steers; the driver keeps (or takes) the pedals. This lets you use openpilot's lateral everywhere, including situations where you don't want it managing speed.

*(personal idea; the safety-layer concept follows the community's MADS work)*

## How it works

A dedicated state machine (`AolDriver` + `AolStateMachine`) runs alongside selfdrived's main one. When AOL is active, controlsd consults `aol.active` instead of `selfdriveState.active` to decide whether to send steering commands, while the main state machine continues to own longitudinal engagement, alerts, and safety events. AOL state is published on its own cereal service, rendered in the UI (steering-active indication without full engagement), and — critically — mirrored into the **panda safety layer**: the safety code has to permit lateral-only actuation, so this branch overrides the `opendbc` and `panda` submodules to the SpysyWeeb forks carrying the real AOL/MADS safety-mode changes. Lane-change alerts that stock openpilot only shows while fully engaged are surfaced during AOL-only steering too, and edge cases (ignition-on phantom engagement, cruise-available edges at boot) are latched out.

## What changed

- `openpilot/spysypilot/aol/` *(new)* — `aol.py` (the `AolDriver` state machine and alert plumbing), `state.py` (`AolStateMachine`, active/enabled states), `helpers.py` (Hyundai always-allow detection).
- `openpilot/selfdrive/selfdrived/selfdrived.py`, `controlsd.py`, `controlsd_ext.py`, `card.py` — AOL driver wired beside the main state machine; lateral-allowed decision routed through it.
- `openpilot/cereal/custom.capnp`, `log.capnp`, `services.py` — AOL state message/service.
- `openpilot/selfdrive/pandad/` (`panda.cc/h`, `pandad.cc`) — safety-param plumbing so the panda knows AOL mode.
- `openpilot/selfdrive/ui/` (`ui_state.py`, `augmented_road_view.py`) — steering-active-without-engagement rendering.
- `.gitmodules` + submodule pointers — `opendbc` and `panda` point at the SpysyWeeb forks (branch `Spysypilot`) carrying the AOL safety-mode code.

> Note: this branch is a collaborator's area of the fork — changes land here on request only.
