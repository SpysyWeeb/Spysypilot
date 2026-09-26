# smooth-stops

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: in progress; awaiting owner field validation.** This branch now carries only radard's low-speed lead gates.
Watch the last metres of a stop behind a stopped car or at walking pace for a close radar return becoming the lead.

## What it does

- **radard** — the unconfirmed low-speed lead override (a radar-only track below the stationary speed threshold)
  needs a track age of 20 cycles (~1 s, `LOW_SPEED_LEAD_MIN_CNT`) and a distance past `max(0.75 m, 0.6 s × v_ego)`.
  A track's age restarts whenever the radar drops it for even one cycle.

*(personal idea)*

## Where the stop landing went

This branch used to own the last 0.3 m/s of every stop inside LongControl (the "thin handoff"): the hold clamp waited
for the car to slow to 0.10 m/s, a 0.15 m/s² "kiss" bounded the planner's request from below while the car rolled out,
and a dropped stop bit with the plan still braking lifted the hold only after 0.5 s. Stock openpilot clamps the frame
the planner's stop bit sets (v < 0.3 m/s) and ramps the brake command toward −2.0 m/s² while the car still rolls; that
ramp was the "headbang" this branch was written to kill. (Before the thin handoff of 2026-08-29, an entry-anchored
taper, a lead floor, a queue-aware anti-creep ratchet and a radarState subscription in controlsd lived here; they never
triggered in the field and re-derived lead physics the planner already owns.)

Since BLoTv3's braking-audit step 1 (`docs/BLoTv3.md` D37, on the `BLoTv3` and `combo` branches) the **planner owns the
stop bit's timing**: it withholds the bit while its landing's kiss lands a rolling car and raises it at the standstill
speed, 0.10 m/s (the Palisade's own standstill flag rises at 0.07–0.11 m/s), or at once when a target is nearer than the
kiss can stop short of. The same decision had two owners, so the LongControl half is retired: `longcontrol.py` is
upstream's again and its clamp follows the plan's stop bit. What it measured before going (the merged `combo` tree,
LongControl variants on the same planner):

- **The speed wait and the settle.** With the hold on the stop bit the settle can no longer run. The wait only mattered
  where the bit rises above 0.10 m/s outside the kiss: 6 of 161 maneuver runs, and the two BLoTv3 cases it failed — the
  plant's check that the stop bit hands the car to the stopping ramp, and a target appearing 0.3 m ahead at walking pace,
  which with a car braking 0.3 m/s² short of the request the settle followed into contact (−0.11 m) where the stopping
  ramp stops 0.12 m clear. On 21 replayed routes (438,623 engaged frames, open loop) it changed 6 stops, only one by more
  than 0.05 m/s²: a queue creep on route 0x3e whose plan never braked past the kiss, so no landing formed and the car gets
  the stopping ramp from 0.26 m/s. The six traced stops closed loop through three ESP fits: 18 of 18 runs byte-identical.
- **The release debounce.** On the same routes it released the hold later than the stop bit on 70 of 104 releases. 65
  were launches: the planner drops the bit on the frame its landing hands the kiss back (−0.05 m/s² there, positive a
  frame later), and waiting for a positive plan held StopReq 50–250 ms into the green. 4 were releases with the plan at
  −0.15…0, held the full 0.5 s. 1 was a 0.25 s stop-bit flicker at rest that it hid. With BLoTv3's step 1b on top: 74
  launches, 5 full holds, 3 flickers — all three the lead-departure pre-release releasing and taking it back 0.2–0.25 s
  later, which is the pre-release's decision to get right, not LongControl's.

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

- **2026-09-26 — the LongControl half is retired.** BLoTv3's planner now owns when the stop bit rises (see "Where the stop
landing went"), so `smooth_stops.py`, its tests and the handoff's tests in `test_longcontrol.py` are deleted and
`longcontrol.py` is upstream's again. The module's claim that the standstill flag asserts at ~0.6 m/s was wrong: the
Palisade's rises at 0.073–0.111 m/s (median 0.098 over 98 edges).

- `openpilot/selfdrive/controls/radard.py` — the unconfirmed low-speed lead override's track-age and distance gates.
- `openpilot/selfdrive/controls/tests/test_radard_low_speed_lead.py` — its tests.

History note: this branch once also carried Smooth Approach / Smooth Release wrappers for braking further out; those were retired in favor of the [`BLoT`](https://github.com/SpysyWeeb/Spysypilot/tree/BLoT) supervisor, which drives the MPC's own knobs instead of wrapping its output. The stop-landing piece followed on 2026-09-26 (above); radard's gate is what remains.
