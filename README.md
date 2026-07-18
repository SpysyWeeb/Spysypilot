# SOL — Sometimes-On-Lateral

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded with [Claude Code](https://claude.com/claude-code), is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

> Formerly known as **Always-On-Lateral (AOL)** — the community-standard term — but renamed because the steering isn't literally always on; it's on when *you* toggle it. The code identifiers (`aol`, `AolDriver`, the cereal `aol` message, the `aol.h` safety layer) keep the original name.

## What it does

**Decouples steering from cruise control.** With SOL, openpilot's lateral and longitudinal control are each usable without the other: press cruise MAIN and the car steers while the driver keeps (or takes) the pedals — or run op long without op steering by leaving SOL toggled off. This lets you use openpilot's lane keeping everywhere, including situations where you don't want it managing speed, and vice versa.

*(personal idea; the safety-layer concept follows the community's MADS work)*

## How it works

A dedicated state machine (`AolDriver` + `AolStateMachine`) runs alongside selfdrived's main one. When SOL is active, controlsd consults `aol.active` instead of `selfdriveState.active` to decide whether to send steering commands, while the main state machine continues to own longitudinal engagement, alerts, and safety events. SOL state is published on its own cereal service, rendered in the UI (steering-active indication without full engagement), and — critically — mirrored into the **panda safety layer**: the safety code has to permit lateral-only actuation, so this branch overrides the `opendbc` and `panda` submodules to the SpysyWeeb forks carrying the real SOL/MADS safety-mode changes. Lane-change alerts that stock openpilot only shows while fully engaged are surfaced during SOL-only steering too, and edge cases (ignition-on phantom engagement, cruise-available edges at boot) are latched out.

## What changed

- `openpilot/spysypilot/aol/` *(new)* — `aol.py` (the `AolDriver` state machine and alert plumbing), `state.py` (`AolStateMachine`, active/enabled states), `helpers.py` (Hyundai always-allow detection).
- `openpilot/selfdrive/selfdrived/selfdrived.py`, `controlsd.py`, `controlsd_ext.py`, `card.py` — SOL driver wired beside the main state machine; lateral-allowed decision routed through it.
- `openpilot/cereal/custom.capnp`, `log.capnp`, `services.py` — SOL state message/service (named `aol` on the wire).
- `openpilot/selfdrive/pandad/` (`panda.cc/h`, `pandad.cc`) — safety-param plumbing so the panda knows SOL mode.
- `openpilot/selfdrive/ui/` (`ui_state.py`, `augmented_road_view.py`) — steering-active-without-engagement rendering.
- `.gitmodules` + submodule pointers — `opendbc` and `panda` point at the SpysyWeeb forks (branch `Spysypilot`) carrying the SOL safety-mode code.

> Note: this branch is a collaborator's area of the fork — changes land here on request only.
