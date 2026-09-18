# smooth-stops

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: in progress; awaiting owner field validation.** Watch the last few centimetres of each stop for any residual creep the kiss doesn't fully arrest, and the release for hesitation on a green.

## What it does

Kills the **"headbang" at the end of every stop**. Stock openpilot jumps to the `stopping` state the moment the
planner's stop bit sets (v < 0.3 m/s) and ramps the brake command to −2.0 m/s² *while the car is still rolling*. This
branch waits for the car to actually slow to 0.10 m/s before that clamp lands, and in between keeps the planner's
request bounded from below by a gentle "kiss" of braking so the stop completes.

This branch has no anti-creep of its own: if the car stalls under the kiss and stops making progress, nothing here
presses harder. The planner and the Palisade's own ESP, which brings the car to rest below ~0.3 m/s whatever is
requested, finish the stop. On the `combo` branch the planner-side landing corridor (BLoTv3) owns anti-creep instead;
against the stock planner this branch runs on, that job is the ESP's.

*(personal idea)*

## How it works

`SmoothStopController` inside longcontrol, at control rate:

- **Hold only once stopped** — the clamp arms once the car has actually slowed to ≤ 0.10 m/s (`STANDSTILL_SPEED`).
  The car's own standstill flag is not used for this: it asserts at ~0.6 m/s, far too early to be believed.
- **The landing** — while the planner's stop bit is set and the car still rolls, the command is the planner's request
  bounded from below by the kiss (0.15 m/s², `STOP_KISS_DECEL`), jerk-limited at 2.5 m/s³ (`SETTLE_JERK`) in both
  directions. Braking harder than the kiss is not held back by that floor, but it still rides the jerk limit, and
  the PID is held reset for the whole landing.
- **Release** — the frame the stop bit drops with the plan asking to move (`a_target > 0`) lifts the hold at once;
  launch response time isn't for sale, so there's no debounce on that edge. A dropped bit with the plan still braking
  is a flicker unless it lasts 50 control frames (0.5 s, `HOLD_RELEASE_FRAMES`).
- **radard** — the unconfirmed low-speed lead override (a radar-only track below the stationary speed threshold)
  needs a track age of 20 cycles (~1 s, `LOW_SPEED_LEAD_MIN_CNT`) and a distance past `max(0.75 m, 0.6 s × v_ego)`.
  A track's age restarts whenever the radar drops it for even one cycle.

What used to be here — an entry-anchored taper, a lead floor, a queue-aware anti-creep ratchet with hysteresis and
radar-dropout grace, and a radarState subscription in controlsd — never triggered in the field and re-derived lead
physics the planner already owns, so it was removed.

## What changed

- **2026-08-30 — radard's low-speed override gets a distance floor.** An unconfirmed radar return closer than 0.6 s of travel can no
longer become the lead: on route 0x2a a stationary return the car drove over (tracked from 3 m to under the bumper) became the lead at
1.1 m the moment it passed the 1 s age gate and put −2.6 m/s² into a landing. Below the stock 0.75 m floor nothing changes.

- **2026-09-02 — the settle's stall ratchet is retired.** BLoTv3's landing corridor now closes the loop on the car's measured
acceleration (its anti-creep press), so the handoff keeps only the kiss (0.15 m/s², the corridor's own number) and the deferred
clamp; two ratchets had been stacking on one output.

- **2026-09-04 — the hand-off is back at 0.10 m/s, and the release is debounced.** The 0.05 hand-off of 2026-09-02 was wrong: below ~0.06 m/s the ESP fades its braking whatever we ask, then coasts ~1 s after StopReq before its clamp bites, so route 0x4b's stops were 1.5 s of fade plus 1.4 s of coast — "a complete stop, then creeping". At 0.10 the clamp lands after 4 cm and 0.8 s (25+ stops on routes 0x33–0x3e). The release is latency-free and flicker-proof: the frame the stop bit drops with the plan asking to move lifts StopReq at once; a dropped bit with the plan still braking is a flicker unless it lasts 0.5 s (route 0x4b t=299 toggled StopReq four times in 0.4 s). Awaiting the owner's drive.

- **2026-09-17 — audit cleanup, no behavior change.** The README is back in line with the code above. The module docstring and
constant comments were trimmed to upstream style; the no-op `reset()` and the unused `v_ego` parameter of `settle()` were removed.
radard's comments were fixed: the age-gate comment was spliced mid-sentence, and the claim that vision-confirmed leads bypass the
low-speed override was wrong — a closer qualifying track replaces the lead either way, vision-sourced or not; a track's age
restarts on a one-cycle radar dropout, not just a sustained loss. Tests moved onto `OpenpilotTestCase` with short names, and gained
cases for the age boundary, the 25 m ceiling, the speed off-switch, a vision lead versus a closer track, the release counter
re-arming on hold entry, and the PID staying reset through the landing. No constant or behavior changed.

- `openpilot/selfdrive/controls/lib/smooth_stops.py` — `SmoothStopController`: hold arm/release and the landing settle.
- `openpilot/selfdrive/controls/lib/longcontrol.py` — the stopping-state transition and hold release route through the
  controller; the pid branch lands the car while the plan's stop bit is set.
- `openpilot/selfdrive/controls/radard.py` — the unconfirmed low-speed lead override's track-age and distance gates.

History note: this branch once also carried Smooth Approach / Smooth Release wrappers for braking further out; those were retired in favor of the [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT) supervisor, which drives the MPC's own knobs instead of wrapping its output. This branch is now the stop-landing piece only.
