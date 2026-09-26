# BLoTv3 — design

**Status: in progress. Not field validated. Do not mark complete before owner field testing and
explicit approval.** Phases 0–3 implemented on `BLoTv3` and replay-gated against BLoTv2; merged into `combo` on
2026-08-29 (phase 4a, replacing BLoTv2) for the owner's field test. Phases 4b–4d (smooth-stops
re-authoring, curve-limiter composition review, SOL/AOL hook review) follow the field verdict. Step 1 of the 2026-09-24
braking audit (D32–D38, §8 2026-09-25) is built and replay-gated, field test pending; the audit's steps 2–5 wait for the
owner's drive.

BLoTv3 restructures [BLoTv2](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2) from `stock`.
It keeps BLoTv2's tuned behavior where BLoTv2 was right and fixes the verified defects; it does
not add tuning knobs or toggles. Standalone BLoTv3 runs on the stock opendbc pointer
(`ACCEL_MAX = 2.0`); `combo` runs the same code on the fork opendbc/panda lineage (4.0 m/s²).

## 1. Sole-owner contract

| Decision | Owner |
|---|---|
| Radar/model lead selection | `radard` and the driving model (stock, not edited) |
| Usable lead, lead presence, model-lead anchoring | `longitudinal_lead.py` (`LeadObservation`, `lead_present`, `relevant_lead`, `anchor_model_lead`) |
| Model stop intent classification (stateless) | `stop_helpers.py` — one copy of every stop constant |
| Effective Chill/Experimental mode | `selfdrived` via `conditional_experimental_mode.py`; owns nothing about stop points |
| Stop shaping, commitment, hold through standstill, release | `force_stops.py` (plannerd) |
| MPC response cost and dynamic headway | `necessity_supervisor.py` (`NecessitySupervisor`) |
| Lead trajectory and obstacle optimization | stock Acados MPC; `long_mpc.py` sets weights once per frame |
| Cruise acceleration target and final arbitration | `longitudinal_planner.py` |
| Landing bound on every stop's last metres, and handing back what it held when it ends | `stop_landing.py` (plannerd), applied after the arbitration |
| The stop bit through a landing | `longitudinal_planner.py`: withheld while the landing's kiss lands a rolling car, raised at the standstill speed (D37) |
| Stopping ramp and standstill handoff below the stop bit | stock `longcontrol.py`; `smooth-stops` in `combo`, whose settle no longer runs during a landing (D37) |
| Vehicle command limits / safety ceiling | opendbc / panda |

Only `selfdriveState.experimentalMode` crosses processes. Stock never had BLoTv2's
`SelfdriveState.conditionalStop{Qualified,Distance,ModelMonoTime,Latched}` fields, so this branch adds
nothing to `SelfdriveState`; on `combo` those fields move into a `deprecated` group at integration so
their ordinals stay reserved. `LongitudinalPlanSource.stop` is added.

## 2. Owner decisions (2026-08-29)

| # | Decision | Ruling |
|---|---|---|
| D1 | Architecture | mode in selfdrived, everything else in plannerd, shared stateless classifier |
| D2 | Turn budget | **removed after the 2026-08-29 field test** ("accelerating out of a curve feels held back"): cruise acceleration is bounded by the envelope alone; `curve-speed-limit`'s limiter and torque veto remain combo's only curve mechanism |
| D3 | `STOP_DISTANCE` | keep 7 m (owner prefers the extra distance); documented fork change **Cross-reference (2026-09-25, not a change to this row):** D34's lead requirement stops the car on this 7 m too, and with D37's measured-speed seed the MPC column now rests the car on it: in the landing test bed lead stops rest 0.32–1.27 m farther back than before (1 of 30 cells within 0.1 m), where the 4 m form rests nearer the previous point (8 of 30) with a softer wheel stop. The geometry is the owner's to revisit (D34) |
| D4 | Acceleration-change cost through standstill | keep BLoTv2's behavior (cost stays on). Owner requirement: launches start smooth but grow quickly — acceptance metric: from a no-lead standstill launch, commanded acceleration reaches 50 % of the envelope within ~1.0 s with no dip; tune the low-speed cruise jerk or the supervisor launch response if not, never by removing the cost. Measured with the real MPC behind a departing lead: first step 0.13 vs 0.40 m/s² per frame with the cost off, half of peak at 1.35 s vs 0.70 s — the owner judges this in the phase-2 field test |
| D5 | Third model lead ("ponytail") | delete (owner never felt it act) |
| D6 | Supervisor lead speed | filtered `vLeadK`/`aLeadK`; MPC keeps raw `vLead` as stock; documented in one place |
| D7 | Supervisor stand-down → FCW alert | no; FCW keeps stock's form |
| D8 | Branches | `force-stops` retired 2026-09-03 after its README was folded here; its sync entry was removed before the ref was deleted. `BLoTv2` remains the frozen replay reference; the radard low-speed age gate stays owned by `smooth-stops` |
| D9 | Module names | `longitudinal_lead.py` (kept — combo imports it), `necessity_supervisor.py`, `stop_helpers.py`, `force_stops.py`, `conditional_experimental_mode.py` |
| D10 | Hold release fallback | 4 s window in which ≥ 80 % of model frames show no stop tier, terminal speed ≥ 1 m/s and the stop corridor is lead-free — positive "clear" evidence, not absence |
| D11 | `selfdrived.py` on combo | collaborator (SOL/AOL) area; ask before phase 4d touches it; the CEM hook lands around the AOL calls without moving them |
| D13 | Committed approach profile (2026-08-29 field test 2) | Force Stops publishes its own plan candidate while a commitment is moving: the constant deceleration that lands `PROFILE_LANDING` short of the point, entered from the car's current deceleration at `PROFILE_JERK`, capped at `PROFILE_MAX_DECEL`, faded out between `PROFILE_HANDOVER_SPEED` and `PROFILE_FADE_SPEED` so the MPC column's own easing and the hold land the car. The MPC's quadratic stop column cannot be front-loaded (route 24: +1.45 → −1.86 over 1.5 s after a commit that needed 1.9 m/s²); the owner's own stops reach the needed deceleration within a second, hold it, ease off **Entry reversed by D37 (2026-09-25; the row is kept as written):** the profile's first value now comes from the target the planner published last frame, `min(aTarget, 0)`, not from the car's measured deceleration — `aEgo` lags that target and rings, so an entry from it stepped the plan on the commit frame (0x5e t=147.0: −0.55); from there it slews at `PROFILE_JERK` on its own value, as before. Whether a stop should be front-loaded at all is the target-shape question left to the owner (F014, step 4) |
| D14 | Change-cost anchor on obstacle handoff | the MPC refills `a_prev` with the current acceleration whenever the binding obstacle column changes (lead0/lead1/stop), not only when an adaptive lead0 policy ends; a committed stop used to inherit the free run's accelerating solution **Cross-reference (2026-09-25, not a change to this row):** "the current acceleration" was the MPC's seed `x0[2]` — the previous published target, which is the plan read `action_t` ahead, not a measured acceleration. Since D32 the refill re-anchors on the target the car was told (`a_told`, the previous published target), which a hand-over also starts the new plan from, while between hand-overs the MPC continues its own plan (D33); D32 keys the refill on the binding obstacle's identity instead of its column and adds the iterate re-seed this row lacked (the first solve after a commit was still linearised on the free run) |
| D15 | Commit speed | `QUALIFY_S` 1.0 → 0.3 s (world-fixed endpoint under strict evidence); a raw lead still blocks a *new* commitment, but only a *tracked* lead (`lead_filter` above `LEAD_GATE`) breaks an existing one — one radar frame reset a red-light commitment 0.5 s before the driver braked, and a flickering lead must not mint commitments between its frames |
| D18 | Landing taper (field test 4) | the profile never plans to reach its landing sooner than `PROFILE_MIN_TIME` (1 s): the need is at most v/2 near the end and only eases, instead of blowing up as the landing closes (route 27 t=1052, −2.4 m/s² at 2.6 m/s) |
| D19 | No speed cap once committed | `v_cruise_cap` is `NO_CAP` while forcing: the shaping cap's cruise floor used to land the car at −1.2 down to walking pace once the profile had faded (route 27 t=1053); the profile, the MPC column (which eases −0.9 → −0.4) and the hold own a committed stop |
| D20 | e2e against a committed profile | while the profile is moving the car, the model's own request joins the arbitration only if it is more urgent by `E2E_STOP_MARGIN` (0.5 m/s²): its late ramp used to overtake the flat profile through `min()` and put the heavy braking back at the end (route 27 t=250) |
| D21 | Green release and lane changes | a path longer than `RELEASE_OPEN_LENGTH` (30 m) for `RELEASE_OPEN_FRAMES` (3) releases a hold at once (saves ~0.2 s of the ~0.5 s the filtered release took; a one- or two-frame flash, route 27 t=263, does not); a lane change (`meta.laneChangeState`) drops shaping and a moving commitment so the stop re-qualifies on the new lane's endpoint (route 27 t=379: the through lane's line held a stop 15 m short of the left-turn lane's) **Audit note 2026-09-02:** `lane_changing()` looked the enum up under `LateralPlan` inside a try/except and so returned False on every frame — the lane-change re-qualification was inert from 2026-08-30 until the audit pass fixed the path (`log.LaneChangeState`) and added the test **Cross-reference (2026-09-17, not a change to this row):** this release also requires no stop evidence on those frames — the guard D28 gave the moving commitment, which the hold's own counter never had (§3, `force_stops.py`) |
| D22 | Landing law for every stop (2026-08-30, corridor form after route 28) | the planner bounds the arbitrated target through the last metres of any stop with a corridor: allowed braking `landing_bound(v)` = 0.70·v + 0.30 above 0.5 m/s (1.35 m/s² at 1.5 m/s, 0.65 at 0.5) and a floor `landing_floor(v)` of 0.40 at 1 m/s (fading to nothing by 1.5 m/s: a queue is not held to a stop's floor), **both tapering to one 0.15 m/s² kiss by 0.15 m/s** so the wheels stop under a whisper. The bound only removes surplus braking: a lead within `LEAD_FULL_AUTHORITY` (5 m) lifts it (the floor stays), the braking that stopping `LEAD_LANDING_GAP` (4 m) behind a lead needs always passes (`total_decel_requirement`), and a watchdog shifts the corridor toward more braking at 0.15 m/s² per second once the car has not slowed for 1 s while rolling. **The landing latches**: it starts on stop intent with the plan braking, lasts through the MPC's hover around zero and through standstill, and ends only on the planner's own release (the lead-departure pre-release, a hold release) or the raw plan positive for `LAUNCH_FRAMES` (3) in a row; the stop bit follows the landed target while landing. Why: routes 23–27 — lead-free landings exceeded the bound in 32–66 % of their last-3 m/s frames, up to 1.1 m/s² at walking pace; e2e, MPC-column and cruise-floor landings had no law. Route 28 (first form, a switched flat 0.40 creep floor): behind a stopped lead the MPC column lets go of the brake by 0.2 m/s and hovers ±0.15 around zero; the floor's on/off edge, fed back through the MPC's starting acceleration, made the target alternate −0.40 / +0.1 every frame for the last half second (8 of 14 stops), the positive frames dropped the stop bit into LongControl's raw PID branch (+0.13 in one frame, `SCC12 aReq`) and once released the hold clamp — the ESP's own accelerometer shows the brake–blip–clamp as a 1.3–1.55 m/s² swing in 0.2 s. This is the owner's original Smooth Stops design (sunnypilot `smooth-stops-dev` v01–v13, June 2026) rehomed in the planner, now with its "release-then-clamp" half — the June LongControl ramp eased toward a settle deceleration before standstill; here the planner's own corridor does it **2026-09-04:** a launch frame never starts a landing — the entry path ignored `launch`, so a lead-departure pre-release ended the latch and the entry re-armed it on alternate frames, toggling the stop bit and StopReq (route 0x4b t=299) **Cross-references (2026-09-17, not a change to this row):** the bound's elbow and the kiss speed quoted above are pre-D24 — D24 moved `KISS_SPEED` to 0.40 and the elbow to 0.9 m/s (`BOUND_BP`); and "a hold release" here means the commitment ending, not the hold bit dropping (§3, `stop_landing.py`) **Cross-reference (2026-09-25, not a change to this row):** D34 deletes the 5 m `LEAD_FULL_AUTHORITY` lift, the 4 m `LEAD_LANDING_GAP` (a lead's requirement now stops the car `STOP_DISTANCE` behind it, D3) and the watchdog (one anti-stall integral replaces it and D27's press); the corridor's tables now describe the car's deceleration against its speed and are requested one actuator delay early; D38 hands back whatever the landing held when it ends, at the cruise jerk; and D37 withholds the stop bit while the kiss lands a rolling car, so "the stop bit follows the landed target while landing" holds only at the standstill speed |
| D23 | Release lift in the landing (2026-08-30, route 0x2a) | the ESP follows a braking increase with ~0.2 s but a release with ~0.7 s (request vs measured over 50 s of low-speed braking, gain 0.99 at steady state), so through every landing the car brakes harder than the plan asks — 0.6 m/s² typically, 1.0 at worst — and a two-frame spike (a radar return under the bumper adopted as the lead) became a second of −1.9. The landing closes the loop on the measured acceleration one way only: when the car decelerates more than the plan wants, the request is lifted by `RELEASE_GAIN` (0.5) of the surplus beyond `RELEASE_DEADBAND` (0.1), at most `RELEASE_LIFT_MAX` (1.0), never above the floor and never above the lead's own requirement (a plan already braking less than that is left alone). Only while rolling and braking. This is the owner's accelerometer idea ("it's about the g-force and how quickly it shifts") in the planner; the maneuver plant gained an asymmetric actuator (`actuator_lag=(0.2, 0.7)`) so landings can be judged through the car's real response **Superseded by D34 (2026-09-25; the row is kept as written):** the lift is deleted. It was an outer proportional loop on the measured acceleration against the raw plan — ×1.5 on every hand-over while active, a 10 Hz ring with the MPC's seed, blind while the bound held. The surplus it answered is lag on a ramp (about +0.02 − 0.54 s × the request's slope, near 0 under a flat request), not the fixed 0.6 / 1.0 quoted above; its purpose moves to D34's feed-forward delay advance. The plant's actuator tuple is replaced by the CAN-identified actuator (D36) |
| D24 | The kiss arrives early (2026-08-31, route 0x2b) | the final blip is the body's pitch return, and it begins while the command is still flat at the kiss: the car carries −0.34…−0.42 of measured deceleration at 0.15 m/s because the ESP releases ~0.7 s behind the request. `KISS_SPEED` 0.15 → **0.40** (the corridor reaches the kiss one release-lag before the wheels stop; bound elbow moves to 0.9 m/s), the release lift's deadband 0.1 → 0.05, and a climbing plan ends a landing only while rolling (`v > KISS_SPEED`) — at standstill the MPC's hover can drift positive for a few frames and the launch authority there is the planner's own release. The maneuver plant grew the asymmetric actuator (`actuator_lag=(0.2, 0.7)`) and its stop-bit stand-in became the thin handoff's settle (`min(plan, −0.12)`) instead of a flat −0.5 that overwrote exactly the behavior under test. Cost: the last ~0.4 m/s is a slightly longer soft crawl **Cross-reference (2026-09-25, not a change to this row):** D34 keeps `KISS_SPEED` 0.40 and the 0.9 m/s elbow as the car-side corridor without re-measuring them as car-side targets (the owner's open item); the lift's deadband went with the lift; the plant's actuator tuple and the `min(plan, −0.12)` stand-in are replaced by the CAN-identified actuator and stock LongControl (D36) |
| D25 | Lead departure releases at 0.5 m/s (route 0x2b t=1540) | a lead that crept at 0.65 m/s and stopped again cycled the hold (StopReq off and back on under a standing car); the pre-release's instant threshold rises 0.3 → 0.5 m/s, slower creeps go through the existing 0.2 s confirmed path |
| D26 | CEM search release (route 0x2b t=1291–1317) | Experimental stayed on 17.5 s past a passed red light: lone 0.70-confidence hint frames refreshed the 4 s intent hold 35+ times while the model's own path showed the road open twice, and the model's e2e braking took 9.3 m/s of speed until the owner's gas tap. Two changes in `conditional_experimental_mode.py`: while active, only tiers at `STOP_HOLD_MIN_CONFIDENCE` (0.8: early/strict/direct) refresh the hold — a lone hint may support entry but not sustain the search; and `CLEAR_CANCEL_S` (0.75 s) of sustained open road (the model's long, moving path — positive evidence the stop is behind us) cancels the hold outright, the ordinary clear hysteresis finishing the exit. Replay gates: route 0x2b's episode 17.5 → 3.6 s with the phantom braking window gone; route 27 diverges from the field only in five short exit tails (12.9 s total), never adds the mode, loses no stop (the two uncovered stops were the owner's own pedal windows in the field) |
| D27 | Anti-creep press (route 0x2c t=727) | driveline creep beat the −0.15 kiss on a slow entry: the car bottomed at 0.22 m/s, re-accelerated for a second with the brake light off, and only LongControl's stall ratchet caught it ~2 s later (3.9 s from 0.5 m/s to standstill vs the 1.0–1.3 s norm). Below `KISS_SPEED`, a measured shortfall (`a_ego` above the target) presses the whole corridor toward more braking immediately: `CREEP_PRESS_GAIN` (1.0) of the shortfall beyond `CREEP_PRESS_DEADBAND` (0.05), at most `CREEP_PRESS_MAX` (0.5). The press relaxes as the car slows, so the wheel stop keeps the kiss **Superseded by D34 (2026-09-25):** the press read the raw plan (off on a hover, on with no creep), had no standstill gate and split one decision with the stall watchdog; one anti-stall integral replaces both |
| D28 | A green releases a moving commitment (route 0x2c t=1105/1135) | the moving-commitment release required the filtered detector below 0.30 AND the 4 s position hold — re-armed by every detected frame, including noisy path dips after the road had opened — so the committed profile kept braking 1.2–1.7 s past the green (6.3 and 4.2 m/s of speed lost post-green) until the owner's gas ended it. A path longer than `RELEASE_OPEN_LENGTH` with no stop evidence for `RELEASE_OPEN_FRAMES` (3) now releases a moving commitment at once — the same release the hold got in D21; the rolling green's remaining latency is CEM's own exit (~1.1 s) plus the cruise ramp (~0.8 s) **Cross-reference (2026-09-25, not a change to this row):** the 30 m length is the hold's standstill calibration; a moving commitment forms with the path end 40–65 m out, so at approach speed the length is always met and this release reduces to three frames with neither `shouldStop` nor strict evidence. It fires when the plan's terminal speed hovers across strict's 1.0 m/s while the plan still stops at the same point: false on 6.5–12 % of moving commitments in replay (0x62 t=3151.56, §8 2026-09-25). F011's plan-physics green was built and held back (D35) |
| D29 | Pursuit tail: after excess braking behind a lead that is already accelerating away, the supervisor keeps the low jerk cost for `PURSUIT_TAIL_S` 3 s after the recovery trigger disarms (route 0x3b t=392: the trigger dropped at the plan's zero crossing and the stiff cost slowed the pickup by ~0.26 m/s² early); ends when the lead stops pulling; following time untouched | field test pending |
| D30 | The committed point follows the model on confirmed evidence (routes 0x58 t=548, 0x59 t=609, 2026-09-06) | both stops rested 3–4 m past the model's settled endpoint while the day's good stops rest ~1.7 m before it (committed point ~2.5 m before it). 0x58: the commit took the first strict frames' endpoint, 5.8 m long; the model settled within 1 s and sat 5–8 m short of the point for 7 s, but the follow-down waited for `DOWN_SPEED` and reclaimed 2 of 4.4 m. 0x59 (a yellow from 60 mph, −3.3 m/s²): the commit was fine (−1.6) but the forward follow chased a 1.5 s endpoint excursion of +8..+14 m during the braking onset and moved the point 4 m; the model then sat 5 m short for 5.3 s with the car above 3 m/s until the owner braked. `FOLLOW_CONFIRM_S` (1 s of frames past the deadband, drained only by contrary frames so route 25's stuttering drift still counts) now gates the forward follow at any speed and the follow-down above `DOWN_SPEED`; below it the follow-down stays immediate (route 38). Open-loop replay of the committed point relative to the settled endpoint: 0x58 +0.8 → −3.0, 0x59 +2.5 → −2.8, route 25 t=1547 −2.9 → −3.8 (0.9 m given back to the confirmation) **Cross-reference (2026-09-25, not a change to this row):** "gates … the follow-down above `DOWN_SPEED`" understates it: before D30 there was no follow-down above `DOWN_SPEED` at all (the latch was immune to a collapsing endpoint there); D30 added one, gated by `FOLLOW_CONFIRM_S`. Its cost: as the point moves in over the last 3–4 s, the need rises late, by up to about 0.95 m/s² from the commit. Its first field data (0x5e, 0x62) is recorded unjudged in §8 (2026-09-25) | field test pending |
| D31 | The published control state matches the acceleration it was computed with (2026-09-06) | `controlsd` assigned `actuators.longControlState` before running `LoC.update()`, so the state travelled with the previous control frame's decision. The Hyundai interface derives both the stop request (`stopping`) and the standstill-exit `JerkUpperLimit` from that field, so on every stop and every launch the car was told a state belonging to a different acceleration in the same SCC12 message. Published after the update now; the skew was one control frame, 10 ms against the 20 ms SCC14 period | `openpilot/selfdrive/controls/tests/test_controlsd_long_state.py` (the published state on the engage, stop and release frames); field test pending |
| D32 | A hand-over re-seeds the MPC from what the car was told, keyed on what the binding obstacle is (2026-09-25, audit F001/F025; round 2 R7.1) | Cause: D14 re-anchored only the change cost. On a hand-over the MPC's first SQP-RTI step was still linearised on the previous obstacle's plan — at a red-light commit, the fake lead's free run climbing toward `ACCEL_MAX` far past the stop — so the column's first candidate overshot (0x58 t=372.9: −1.92 where the converged answer is −1.26) and rang for about 0.5 s: the commit jolt. The hand-over was also keyed on the argmin column index, so a fake lead turning real in slot 0 got no refill, a lead0 ↔ lead1 label swap of one car refilled, and a column and a stopped lead in the same place refilled on every noise flip. Change (`long_mpc.py`): the hand-over follows the binding obstacle's identity — the free run (fake lead), a real radar lead in either slot, the stop column — which changes only when an obstacle comes or goes: the column's commit and release at once; a radar lead taking over from the free run after `LEAD_PRESENCE_FRAMES` (2: removes the 20 single-frame owner runs in 66,716 frames of lead traces); a lead appearing ahead of the committed column at once; a lead that has gone after `LEAD_ABSENCE_FRAMES` (7, chosen on the outcome of 1–15-frame dropouts behind braking leads in the real planner). A lead and the column trading the binding while both are present is not a hand-over, nor is any real lead → real lead change (a cut-in by another car carries the plan over). The two counts time only the hand-over: the QP plans against the obstacles present each frame, so through a dropout it plans against the free run from the first absent frame, as the shipped build does. On a hand-over the new owner's plan starts from what the car was told — `set_cur_state(v, a, a_told)`, with `a_told` the target the planner published last frame (round 2) — D14's refill re-anchors there, the iterate restarts from the constant-acceleration rollout of that state (at rest once stopped), and that frame runs a second SQP-RTI iteration; between hand-overs the plan continues from D33's seed. `a_next`, the plan one frame on, is D33's seed. The supervisor's policy still follows the lead0 label. Why the told target: the first build restarted from D33's own-plan seed — the plan being discarded — and every lead → free-run hand-over while the MPC drove stepped the published target down 0.37–0.88 in open-loop replay (up to 1.29 in the plant) while the car accelerated away; the shipped build never had that dip, since it seeded the published target every frame. Numbers (replay and plant, nothing driven): all 13 column entries in 9 field windows within 0.001 of converged (were 0.13–0.70 off); the published commit step −0.100 on 9 of 10 field commits (D37's entry), the tenth, 0x59 t=608.8, −0.396, the column's converged need at the car's true speed (step 3); the maneuver red light at 14 m/s and `ACCEL_MAX` 4.0 steps 0.100 with no reversal (the previous code on the same plant: 0.517, 2 reversals); a noisy column/stopped-lead tie 45–59 reversals → 0; lead → free-run hand-overs while the MPC drives, 21 routes open loop: 8 of 8 step up +0.23…+0.27 (shipped +0.19…+0.25 on the same frames), and 0 of 14 plant lead-leave runs collapse; lead-appearance edges better or equal on all six gate routes (0x54 0.151 → 0.081, 0x3e 0.278 → 0.062); the emergency lead-brake windows (0x3a t=513, 0x3e t=713) react within one frame of before; +87 µs x86 on a hand-over frame, +1 µs otherwise. Residuals, recorded for step 3 to re-measure (R7.5): an urgent lead hand-over's first frame lands up to 0.106 (aggressive) / 0.246 (standard) short of converged after the clip, now pinned per personality by the unit tests — a third iteration closes that but lands a 20 m/s commit 0.135 off, so two are kept; a stopped lead appearing ahead of the committed column is never softer than shipped while the stop builds (the first build: up to 0.091), and up to 0.022 softer on its first two frames late in the stop; a cut-in while another lead is in view is up to 0.116 softer on its first frame; the first radar frame of a lead after the free run is solved on the stale free-run iterate, as shipped — in an emergency appearance (contact in every variant) 0.44–0.52 m further in than an immediate re-seed, kept as the price of phantom filtering; a lead back within 3–6 absent frames resumes its stale plan (up to 0.66 softer after 5 frames); a false dropout of 7 frames or more mid-brake hands to the free run from the told target, 0.33–0.53 m less minimum gap than the first build in a feasible brake but never less than shipped (84 cases); an owner change while the plan releases (a commit behind a lead pulling away) is up to 0.226 softer on its first frame than the first build; an owner change at rest with a hard told target (−3.5) plans +3.80 on that frame, as shipped, unreached in the hold and launch probes. Step 3's option: keep the last lead in the QP through the absence hold, so the obstacle and the re-seed change on the same frame. What the owner should feel: no jolt and no ring where a red-light stop commits; a stopped car appearing ahead is braked for from its second radar frame; no chatter where the stop line and a stopped car coincide; at a lead departure no change from the shipped build — the target keeps climbing through the moment the lead is dropped. A lead present for only a few frames, or a one-frame radar distance glitch, is now braked for and released (0x3b t=214.7: −1.09 for one frame and back, about 0.5 each way). What may be felt: a radar dropout of 2–6 frames while braking hard lets off further than shipped on frames 2–6 (the shallowest output 0.27–0.74 higher in a synthetic sweep; frame 1 as shipped; minimum gap within ±0.16 m of shipped), because the QP free-runs from the first absent frame and the own-plan seed follows it until the hold hands over. **Field test pending.** |
| D33 | Bumpless hand-over to cruise, and the MPC seeded with its own plan (2026-09-25, audit F003/F005; round 2: the seed clipped, the landing's own carry moved to D38) | Cause: the cruise candidate slewed on its own value while another candidate — or the landing law's kiss — set the output, so when that owner left (a CEM exit, e2e dropping through the D20 margin, a Force Stops release, the profile ending, a launch from a hold) the output jumped to the stale cruise value or to the free run: +2.29 in one frame at a mode exit in the audit's scenario, +2.58 in a logged drive. And the MPC was seeded with the published target — the plan read `action_t` (0.55 s) ahead — as its current acceleration, so every edit of the target came back at −0.49 on the next frame and rang at 10 Hz, and at standstill the plan mirrored the kiss. Change (`longitudinal_planner.py`): the planner keeps the identity of the candidate that set the output (`plan_winner`: mpc, column, cruise, e2e, profile) and publishes `plan_source`; it no longer overwrites `mpc.source`, which is again the MPC's own column. When the winner leaves the candidate set, cruise slews from the published target instead of its own value, so it carries the hand-over at `J_CRUISE(v)`; it is never anchored on frames where the winner stays (that is the band-aid the finding names — it would hold back the release behind a lead pulling away). The first build also carried a launch ending the landing's kiss here; round 2 moved that to the landing, which hands back whatever it held on every exit (D38). Cruise's stop bit comes from the set speed (`v_cruise − v_ego`), so a cruise candidate carrying a negative hand-over cannot raise one. The MPC is seeded with its own `a_next`, clipped to [`ACCEL_MIN`, `ACCEL_MAX`], while it or its column won the previous frame — also while the landing bounds or hands back what is published from it — otherwise with the published target; a reset seeds the car's `aEgo`. The clip (round 2): the OCP's plan can run past `ACCEL_MIN` (−3.87 at 35 m/s toward a stopped car), and a plan continued from braking the car cannot do lets the brake off. Numbers (unit, plant and open-loop replay): CEM exit +2.29 → +0.073 per frame; an e2e −1.0 exit +2.77…+3.80 → +0.075; a moving Force Stops release +0.56…+1.20 → at most `J_CRUISE`·dt (0.07–0.095); the profile ending under the column +1.16 → +0.088; the seed impulse −0.488, +0.414 → 0.000; the MPC's standstill candidate while it owns the hold +0.15…+0.23 → at most +0.046; the give-back after a lead-brake onset +0.56…+0.81 → +0.06…+0.18 (the rest is the solver's first iterate); the worst let-off while saturated over 12 emergency cases 0.064 / 0.076 at `ACCEL_MAX` 2.0 / 4.0 (0.325 / 0.360 with the seed unclipped; the 27 m/s stopped-car case 0.056 / 0.053 against shipped 0.030 / 0.042), every braking level reached on the same frame as shipped or sooner; the whole of step 1 over 21 routes open loop at 4.0: leave-class up-steps 13 → 0 above 0.3 and 26 → 0 above `J_CRUISE`·dt, alternations above 0.15 189 → 6 on six gate routes and 153 → 13 on 15 held out, and in stop and landing frames alternations above 0.05 391 → 3 and 340 → 10. What the owner should feel: no surge when Experimental drops while braking or when a stop is released — the car eases from where it was at the cruise rate; turning Experimental off while the model brakes now releases at the cruise jerk (about 1.6 s from −2.5 at highway speed, about 1 m/s more speed lost — the gas pedal ends it at once); crisper braking onsets behind a braking lead; no brief let-off during a full emergency brake. What may be felt: when cruise takes over from the MPC just after a lead leaves, the seed switches from the MPC's own plan to the published target and the target can dip for one frame (three synthetic lead-leave runs at `ACCEL_MAX` 2.0, e.g. +1.32 → +0.85 → +1.23; 2 dips above 0.1 on the six gate routes and 4 on the held-out ones, open loop) — neither anchor form tried removed it without changing launches or commits. Both loops are upstream's (#38367); worth reporting there. **Field test pending.** |
| D34 | The landing law without loops: delay-compensated corridor, lead requirement on `STOP_DISTANCE`, one anti-stall integral (2026-09-25, audit F006/F007/F013/F030; supersedes D23 and D27) | Cause: the release lift (D23) was an outer proportional loop on the measured acceleration against the raw plan — ×1.5 on every hand-over while active, a 10 Hz ring with the MPC's seed (impulse envelope 0.90 per frame), blind while the bound held. The 5 m close-lead lift (D22) was a hard authority switch on a radar `dRel` quantised in 0.1 m bins: on 0x54 t=1910.5 the reading sat at 5.0000 with sub-millimetre jitter and the switch flipped on about 30 µm of change, and the MPC seed's ring (removed by D33) carried each flip on for about 0.5 s. The corridor requested at a speed what the car should do at that speed, with nothing for the ~0.55 s the car takes to answer, so the car reached the kiss still decelerating about 0.3 and the body swung a median 0.80 m/s² at the wheel stop. The creep press (D27) read the raw plan (off on a hover, on with no creep), had no standstill gate and split one decision with the stall watchdog. Change (`stop_landing.py`): the lift, the 5 m switch, the press and the watchdog are deleted. The corridor's tables (`landing_bound`, `landing_floor`) now describe the car's deceleration against its speed — the values are D24's, inherited, not re-measured as car-side targets — and the request edges are those tables advanced by the car's `action_t` (`longitudinalActuatorDelay` + `DT_MDL`, 0.55 s on the Palisade, passed in by the planner): bound breakpoints 0.40 / 0.9 → 0.531 / 1.524 m/s, floor 0.40 / 1.0 → 0.493 / 1.178 m/s. The 3.5 m/s ceiling stays, so above the elbow the bound sits above the delay-advanced corridor (+0.21 at 2.0, +0.43 at 2.5, +0.66 at 3.0 m/s). Pure feed-forward: neither edge reads `aEgo`. A lead's requirement always passes: the braking that stops the car `STOP_DISTANCE` (7 m, the MPC's own geometry, D3) behind it in at least `LEAD_MIN_GAP_BUDGET` (0.5 m) of room, and inside that whatever stops it 0.5 m short in the room left after `action_t`, which grows without limit as that room closes; the bound is the larger of the corridor and the requirement. One anti-stall integral: while rolling above 0.10 m/s it integrates `aEgo + floor(v) − 0.05` at 1.0/s, capped at 0.5, and presses both edges; its range fades with the floor between 1.0 and 1.5 m/s; at standstill it is held and not applied. Numbers for this change on the first build (open-loop replay, and a closed-loop test bed over six ESP models including the CAN-identified one; nothing driven): 0x54 t=1910.5 largest step 0.626 → 0.069, 0x5e t=961.6 0.367 → 0.078, the 0x5e t=152.8 latch entry 0.326 → 0.065; landing request reversals of 0.1 / 0.2 / 0.3 or more 91 / 40 / 18 → 20 / 15 / 8 on six routes and 184 / 61 / 32 → 42 / 23 / 11 on 15 held-out routes; the standstill request always −0.15 (before: down to −0.60 in replay); a stopped car appearing 0.8–2 m ahead inside a latched landing ends exactly as with the 5 m switch in all 15 close cases of a closed-loop probe; integral wind-up above 1.5 m/s 106 frames → 0 over 21 routes. Re-measured on the integrated tree (the test bed): red lights on the CAN-identified model, wheel stop median −0.218, mean −0.225, worst seed −0.324 (before −0.324 / −0.380 / −0.681), rest within 0.45 m of before in all four red-light cells, landings 0.42–0.98 s longer; over six ESP models the red-light wheel stop means −0.21…−0.49 and landings a median 0.69 s longer (−0.16…+1.95 s), 3 of 24 cells resting more than 0.5 m from before; request alternation at most 0.033–0.073 per stop (before up to 0.84); with creep torque 180 of 180 stops complete at both creep levels (before: 119 and 102 of 180); lead stops rest 0.32–1.27 m farther back than before (1 of 30 cells within 0.1 m) — the cause is D37's measured-speed seed, which lets the MPC column stop the car on its own 7 m (with it reverted 23 of 30 cells are within 0.1 m). What the owner should feel: walking-pace stops behind a stopped car no longer chatter around 5 m, and the wheel stop is softer; a car or return that appears within a metre or so gets the full braking the plan asks; lead-free red lights let off earlier in the last 3.5 m/s and arrive at walking pace carrying less braking, stopping a little softer and up to about a second later; no lift step at the landing's entry; a launch from a stop always starts from the kiss; a car that creeps under the kiss is pressed and still stops. Watch for a let-off and re-brake late in a stop where the model's late ramp re-joins after a Force Stops let-off (1.75 car reversals of 0.1 or more per stop on the CAN model against 1.88 before: e2e's re-join, F024, which the landing passes; 0.12 per stop on plain red lights, which had none). Open: at creep 0.5 the gate cases re-accelerate up to 0.054 m/s against a 0.05 gate (2 of 36 runs; 15 of 180 creep-0.5 runs above 0.05, at most 0.068; creep 0.3 at most 0.047), untuned — the creep model is the audit's hypothesis from one field event (0x3a t=557, open loop); the 0.10 m/s standstill edge (1 of 1,163 field landings re-crosses it, on the frames the car's own standstill flag toggles); the press drops out in one frame at that edge (for a car braking 0.3 short, `aTarget` −0.418 → −0.150; the stop bit rises on the same frame and LongControl's stopping ramp owns the car there, so the car does not feel it, but the log does); as a lead starts to move inside a latched landing its requirement lets go of +0.10…+0.14 in one frame (0.003–0.042 over `J_CRUISE`·dt in 24 of 192 plant runs, under the 0.15 bound); the requirement reads the raw radar lead, so a one-frame dropout of a close stopped lead lifts the output +0.21…+1.09 for that frame (the 5 m switch had the same exposure; feeding it D32's held lead would close it); a car braking 0.3 m/s² short of every request rests 2.94–2.98 m behind a stopped lead, because the integral closes on the deceleration, not on the gap (a strict expected failure, step 3); `KISS_SPEED` and the elbow as car-side targets, and the 7 m geometry — the 4 m form now rests nearer the previous point (8 of 30 cells within 0.1 m against 1) with a softer lead-stop wheel stop (−0.21 / −0.25 / −0.41 against −0.26 / −0.34 / −0.59 on the CAN, nominal and slow ESP models) — are the owner's to revisit. The landing's own exit behind a creeping lead, open at the first merge (+0.849 at 0x3e t=1111.4), is D38's. **Field test pending.** |
| D35 | A committed stop survives the model's "stop here" (2026-09-25, audit F054; F011 held back) | Cause: at walking pace the model's stopped plan ends a few centimetres behind the car (path end −0.05…−0.30 m) — stop evidence — but `path_end ≤ 0` reset a live commitment, before the standstill hold decision: 13 times on 30 routes of replay, 12 of them moving at 0.21–1.36 m/s (12 of the 29 commitments that reached the landing), each wiping the point, the column and the profile's anchor; the maneuver red light lost its commitment at 0.30 m/s and never held. Change (`force_stops.py`): a non-positive endpoint resets only when nothing is committed, and only after the standstill hold decision. A commitment keeps its point through it at any speed, reading it like any endpoint inside the setback — qualify, the windows, the latch and the green all need a positive endpoint, so none fires; the point dead-reckons within `DOWN_DEADBAND` of the car and is followed in at `DOWN_RATE` beyond (above `DOWN_SPEED` only after `FOLLOW_CONFIRM_S`, D30) — the hold forms from it at standstill, and a re-arm hold (after a gas tap or a lead broke the commitment) forms on it when the model calls the stop. Numbers: drops 13 → 0 in the logged census and 10 → 0 on three held-out routes with Experimental forced; only those episodes change (moving green releases, the 100 holds formed and the 36 hold releases are identical); holds form on the same frame or earlier (by up to 1.7 s); the hold's point is dead-reckoned (−0.77…−2.64 m) instead of re-read (−0.02…−0.46); with the step-1 planner the hold publishes −0.15 on every frame at any of those points and the launch is identical (1 m/s at 1.45 s); 0x5e t=751.1: the release and re-commit at 0.50 m/s are gone and the hold forms on the same frame (stop_x −1.79); the maneuver red lights now end in the hold. At approach speed, unseen in the data (every frame a commitment ran through a non-positive endpoint was at walking pace, at most 1.36 m/s): in a synthetic run, 3 s of zero plans at 8.8 m/s pull the point in 4.1 m and the car rests 6.95 m before the line, where the shipped code dropped the commitment and re-committed (1.22 m); a 0.1–1 s dropout keeps the commitment the shipped code dropped; what an unreachable endpoint means is step 2's estimator's. What the owner should feel: nothing — the difference is under the stop bit, where LongControl's stopping state owns the wheels; the hold no longer depends on a re-qualify at walking pace. **Held back, the owner's call at step 4 (F011):** one plan-physics green for the moving commitment and the hold — release only when the plan, continued past its end as a 1.3 m/s² stop, still runs at 3 m/s or more past the point — built and kept out of the tree. It removes D28's approach-speed false release (5 → 0 above 1 m/s in the census; on three held-out routes with Experimental forced 4 → 1, with all 9 real greens taken against 7 and the median latency 0.10 → 0.00 s), but it reads the model's anticipated launches — plans that stop or crawl at the line, then relaunch inside the horizon — as greens: false releases at walking pace 1 → 3 in sample and 0 → 4 on the held-out routes, the hold's releases there no longer identical (21 → 23), and the traced 0x58 t=381.6 hold lost (in open-loop replay its green launch came 1.6 s late, then stepped +3.03). The question it leaves, first seen on the big model's crawl-through plan at 0x7e: does an anticipated launch count as a green? It is decided at step 4, where the release belongs to the one stop law and the episode, together with CEM's open-road test (`stop_release_open`), which reads the plan the same way. Until then D28's green stays as it is (see its cross-reference) and the 0x62 t=3151.56 false moving green remains; with D37's entry its let-off is as deep as the braking had reached (up to 0.31 in one closed-loop fit). **Field test pending.** |
| D36 | The maneuver plant is the car, and the red lights assert on the plan (2026-09-25, audit F017/F016; round 2 R7.2, R7.6) | Cause: BLoTv3's stop gates ran on a Civic with an ideal actuator, `aEgo` equal to the applied acceleration, exact radar, a stop-bit stand-in (`min(plan, −0.12)`) that matched no LongControl, and a fake model raising `shouldStop` at 3 m whatever its speed; the red-light maneuver sampled only the ends of one approach at one speed and one `ACCEL_MAX`, so the commit jolt (a 0.418 step and 3 reversals at 4.0) passed. Nor could any maneuver see a car that does not do as asked, or driveline creep. Change (`plant.py`, `maneuver.py`, `test_longitudinal.py`): every BLoTv3 maneuver runs on Palisade CarParams under openpilot longitudinal (`action_t` 0.55 s) with stock LongControl at 100 Hz, `aEgo` from the car's own speed filter, radar `dRel` on the radar DBC's 0.1 m grid and `shouldStop` raised as modeld raises it; upstream's maneuvers stay on the Civic. The CAN-identified actuator (per speed band a dead time, a bite and a release time constant and a gain; its dead time and low-speed asymmetry are not identified; held wheels do not wind it) replaces `actuator_lag=(0.2, 0.7)` and drives the lagged landing, the continuity red lights and the new lead maneuvers. The plant car can also fall short of its request while it rolls (`accel_error`: brakes that fall short, and the same push at cruise and on a launch) and creep on its driveline below 1 m/s whenever the request is milder than −0.2 (`creep`) — an assumed model from D27's event, not an identified one; both are inert at their defaults. Red lights at 14/17/20 m/s at both pins, through the actuator: from the call, while above 1 m/s, no one-frame step above 0.15 (3 m/s³), no let-off and re-brake of 0.15 or more before the final ease (a gradual one counts), and Force Stops commits inside that window. At 22 m/s the stop column's converged need at the car's true speed steps 0.208 on the commit frame: a strict expected failure (step 3), whose companion test lets only the commit frame step, braking by at most that need (0.206) plus the MPC's 0.1 first-solve tolerance. The same red lights on an ideal actuator, 14–22 m/s, are plain tests and guard D37's entry. A car braking 0.3 short of every request stops clear of a stopped lead from 10 and 15 m/s — no contact, never nearer than 2 m, the red lights' rule above 1 m/s — and its rest gap against the file's 4.0 m is a strict expected failure (2.94–2.98 m: the anti-stall integral closes on the deceleration, not on the gap; step 3). A stopped lead inching forward or pulling away while the landing is latched, five cases (rolling at 1.0 m/s without creep, at rest without creep, rolling at 0.6 m/s with creep): every frame that starts with the target held below the planner's plan rises at most `J_CRUISE(v)`·dt (D38), judged without reading the law's own flags. forceDecel is judged on the plan, as upstream does. The whole maneuver suite runs at `ACCEL_MAX` 2.0 and 4.0; its expected failures are `unittest.expectedFailure`, strict under pytest and CI's unittest runner alike, with the cause at the marker. Numbers: the previous code on the new plant steps 0.517 / 0.675 / 0.556 / 0.270 with 2 / 2 / 2 / 0 reversals at 4.0 (0.145 / 0.135 / 0.454 / 0.817 and 0 / 0 / 2 / 4 at 2.0) and fails the asserts in 4 of the 6 through-the-actuator cases at 14–20 m/s; step 1 steps 0.100 / 0.100 / 0.100 / 0.208 through the actuator and 0.100 / 0.100 / 0.100 / 0.135 on the ideal one, with no reversal. The lagged stopped-lead landing, red on the first build (excess 0.153 against 0.15, wheel stop −0.341 against −0.25), is green at both pins: excess 0.065, wheel stop −0.178, rest 6.96 m — it needs both D37's measured speed (the excess) and its withheld stop bit (the wheel stop); the thresholds were not raised. The suite catches a revert of D14's refill, of D32's re-seed, of the profile's jerk limit and of its entry (D37), of the lift, of D37's speed seed and stop bit, and of D38 whole or in part; it does not catch D34's delay advance (only its unit test does), the 5 m switch, the anti-stall integral's own creep stop, or D33's seed once D37's speed seed is in (their unit tests do), and it has no queue near 5 m and no CEM-entry case. Test only: nothing to feel, no field gate of its own. |
| D37 | The MPC plans from the car's measured speed, the landing's kiss owns the last metre, and the profile enters from the published target (2026-09-25, round 2: audit F041, F048 and F039's seed half; reverses D13's entry) | Cause: three owners read the wrong state. (1) The MPC's speed state was `v_desired_filter`, an open-loop integral of the published target pulled toward `vEgo` with a 2 s constant. Nothing below the planner closes a loop on speed on this car, so every mismatch between command and car — the ESP's slow release, a car braking more or less than asked, a grade — became phantom speed: +0.20…+0.45 m/s in the last metre of five traced lead-free stops, and a car read slow at a hard braking onset. (2) The stop bit rose below 0.3 m/s while the landing's kiss was still landing the car; stock LongControl's stopping state then ramps at 1 m/s³ toward −2.0 and ignores `aTarget` (the lagged landing maneuver's wheel stop −0.341 against −0.25), and `combo` re-implemented the kiss inside LongControl's settle. (3) The committed profile's first value was `min(aEgo, 0)`: `aEgo` is the speed filter's estimate, lags the target and rings on a step; at 0x5e t=147.0 it read −1.75 against a published −1.26, the profile entered at −1.82 and won the frame — a −0.552 one-frame brake step on the commit (0x62 t=3151.4: −0.202). D32 had removed the MPC's own commit overshoot, which had hidden these. Change (`longitudinal_planner.py`, `force_stops.py`): (1) `mpc.set_cur_state(max(vEgo, 0), …)`; the filter, its reset and its integration are deleted, so `longitudinalPlan.speeds[0]` is `vEgo` — a deliberate departure from upstream's filter seed (#38367), which relies on a tracking loop this car does not have; worth reporting upstream. (2) While the landing's kiss lands a rolling car the plan withholds the stop bit; a Force Stops hold still asserts it. The bit goes to LongControl at once at `vEgo` ≤ `STANDSTILL_SPEED` (0.10 m/s: the Palisade's `carState.standstill` rises at 0.073–0.111, median 0.098, over 98 edges, and `combo`'s field StopReq hand-off at 0.10 lands the clamp after 4 cm), or when a lead is nearer than the kiss can stop short of (the landing's own lead requirement exceeds its bound). Once risen inside a landing it stays while its own rule holds it (below 0.3 m/s), so a rebound or a wheel-speed tick at rest does not hand the car back. (3) `ForceStops.update(…, a_prev_output)`: the planner passes the target it published last frame; the profile enters from `min(a_prev_output, 0)` (a non-finite target reads 0) and slews at `PROFILE_JERK` on its own value from there; Force Stops no longer reads `aEgo`. This reverses D13's entry "from the car's current deceleration" and the 2026-09-17 note that the next commitment "enters its ramp from the car's own acceleration": a joiner starts from the output, F003's rule. F039's other half, the need evaluated one actuator delay ahead, stays with step 3. The audit shipped (1) with D33's seed because a measured speed on the old output seed rang through the lift; with the lift gone (D34) the plant shows no ring either way, and the maneuver suite no longer guards D33's seed (its unit tests do). Numbers (unit, plant, closed-loop replay through a fitted ESP, open-loop replay; nothing driven): (1) the plan's speed error with a car lagging its target 0.61 m/s → 0; a car braking 0.3 m/s² short of every request stops 2.94–2.98 m behind a stopped lead through the CAN-identified actuator, where the filter seed drives into it (−4.6…−7.2 m); 17 closed-loop logged-lead windows: minimum gap median 5.80 → 6.35 m (worst 2.44 → 4.65), and with the car braking 0.3 short median −2.99 m (inside the lead) → +2.69 m; the emergency lead-brake windows reach −3 0.05–0.20 s sooner; the six traced stops' closed-loop plan reversals 22 → 20 over the first build; the lagged landing maneuver's excess 0.153 → 0.065. (2) Over 21 routes the stop bit rises at a median 0.093 m/s instead of 0.288, with no new edge pairs; stock LongControl now equals `combo`'s on the six traced stops × three fits (the largest difference 0.000; the first build's bit put them up to 0.56 apart); the lagged maneuver's wheel stop −0.341 → −0.212 (with (1): −0.178); a 60 s wait with rebound speed readings drops the bit 0 times; a target appearing 0.3 m ahead at 0.28 m/s with a car 0.3 short rests +0.12 m clear. (3) Commit-frame steps on the ten field commits: 9 of 10 exactly −0.100 (0x5e t=147.0 −0.552 → −0.100, 0x62 t=3151.4 −0.202 → −0.100); the tenth, 0x59 t=608.8, −0.396, is the column's converged need at the true speed, the same class as the 22 m/s maneuver's strict expected failure (step 3); the committed point, the hold's stop_x and every Force Stops state identical on every frame; the ideal-actuator red lights 0.169 / 0.183 / 0.180 → 0.100. What the owner should feel: firmer, earlier braking behind a braking lead with more room kept; no extra braking in the last metre of a red-light stop; behind a stopped lead the last 0.3 m/s takes about twice as long (0.65 → 1.20 s on the fitted plant), the car carrying −0.31 instead of −0.52 at 0.3 m/s and −0.17 at the stop, and resting about 1 m farther back, on `STOP_DISTANCE` (7.1 m instead of 6.15 m); on `combo` nothing different at the wheels in the last metre (the plan now carries the kiss the settle carried, and LongControl stays in pid until 0.10 m/s), on standalone BLoTv3 the wheels stop under the kiss instead of LongControl's ramp; a car or person appearing within about a metre while creeping still gets LongControl's stopping ramp at once; no extra brake bump when a red-light stop commits — the braking keeps building from where it was at 2 m/s³, and where the car lagged the plan the committed level arrives 0.1–0.15 s sooner. Costs, recorded: (1) the plan now corrects the car's real deviation, so lead following pivots more: over the 17 windows plan reversals 57 → 63, car 41 → 43, let-off and re-brake sums 6.70 / 4.95 → 8.97 / 6.16 (0x62 t=1152.4 gains a let-off −2.23 → −0.98 and a re-brake to −1.30 behind a lead); two commits now step at the column's real need (the 22 m/s maneuver 0.208, 0x59 t=608.8 −0.396), step 3's; at standstill behind a standing lead the MPC's true-speed hover (+0.10…+0.24) meets the stop bit's 0.1 threshold, so 135 engaged standstill frames in 14 episodes drop the bit in open-loop replay where the shipped build held it (closed loop with a standing lead the bit holds and the car travels 0.00 m), and with σ 0.008 of speed noise a queue that inches gets 8–21 stop-bit edges in 30 s, all the MPC's own bit — F009's cause, not this step's. (2) On `combo` the settle no longer runs during a landing; retiring it is the owner's branch-structure call. (3) A commit made while the published target is positive steps by that target plus 0.1 when the profile wins (+0.3 → −0.400); not in the data (0 of 27 moving commits on 21 routes had a positive target). **Open at merge:** the profile joining from the published target can evict an e2e request that set it: when the model ramps its braking at 2.5–4 m/s³ into a commit, the profile enters within `E2E_STOP_MARGIN` of e2e, D20's gate drops e2e, and it re-joins 4–9 frames later with a 0.56–0.70 brake step (no worse than the shipped build there). A three-line incumbency clause — e2e keeps its place while it led and is at or below the profile — takes that to e2e's own 0.150 / 0.200, changes 0 of 708,319 route frames and removes two closed-loop pivots on 0x62; it is built and measured, not in this tree. Its trade: a model already braking harder than the profile keeps the car, so the stop point follows the model there (up to 8 m shorter in a synthetic constant-request case). **Field test pending.** |
| D38 | The landing hands back what it held (2026-09-25, round 2 R7.6) | Cause: the landing edits the output after `min()`, and on every exit it dropped its edit in one frame — the kiss at a launch, the floor and the anti-stall press at a climbing-plan exit. The first build's planner carried only a launch ending the kiss (D33), so the landing's own climbing-plan exit was uncovered: +0.849 in one frame at 0x3e t=1111.4 in open-loop replay, and in the closed-loop test bed. Change (`stop_landing.py`, `longitudinal_planner.py`): the landing is the one owner of its release. `StopLanding(dt, action_t, release_jerk)` takes the planner's cruise jerk schedule (`J_CRUISE_BP`, `J_CRUISE_VALS`), passed in, not imported. When the latch ends — a launch, the plan climbing for `LAUNCH_FRAMES` above `KISS_SPEED`, or the speed leaving the window — after the landing held the output below the plan (by the floor, the press, or a running release), the output rises from where it was held at `J_CRUISE(v)`·dt per frame until it meets the plan; the floor and the press go back together, so the integral adds nothing to the exit step; a landing that re-latches meanwhile does not drop the release; braking always passes at once. An exit from a frame where the landing passed the plan, or lifted it with the bound or a lead's requirement, carries nothing — the step is the plan's own, and rate-limiting it would be F003's named band-aid; if the winner leaves there, the planner's winner-absent carry (D33) and D32's told-target hand-over take it. A reset drops a release (the planner's reset at disengage or gas, a non-finite speed); a non-finite plan frame passes through and the release carries on from the last finite output. The planner's own landing carry is deleted. While the MPC wins `min()` it keeps continuing its own plan through a release, as while latched: seeding it from the carried output instead brings D33's standstill mirror back (13 standstill releases a frame longer over 21 routes) and bought nothing measurable (1,152 closed-loop runs: plan reversals 588 → 579, car reversals 886 → 903). Numbers (open-loop replay, the landing test bed closed loop with creep, the maneuver plant; nothing driven): 21 routes, 93 latch exits (16 moving): exit-frame up-steps above 0.3 61 → 0 and above `J_CRUISE`·dt 87 → 0 against shipped, largest 0.838 (the first build 0.849) → 0.100; 88 releases over 300 frames, the longest 16; the departing-lead test bed, 1,152 runs at creep 0 / 0.3 / 0.5: 0 runs above `J_CRUISE`·dt (the first build: 41 / 71 / 108 runs, up to 0.849), least gap 4.20 m; the plant grid, 192 runs: 0 above; against shipped, a hold release +0.25 → +0.10 per frame from the kiss and a lead departure from a stop +0.23…+0.31 → +0.10 per frame; a lead departure reaches +0.5 at 1.05 s and +1.0 at 1.85 s on the Palisade plant (the first build 1.00 / 1.85 s); stops that end at standstill without an exit are identical run for run to the first build's law (792 runs). What the owner should feel: when a creeping queue ahead moves off while the car is being landed, the brake lets go at the cruise jerk, not in one frame — the deepest hand-back over 21 routes starts at about −0.43 and takes 9 frames (0x3e t=886.1), and a moving exit's hand-back in the test bed lasts a median 3 frames; launches from a hold and behind a departing lead start from the stop's last whisper of brake (a hold launch about 0.05 s later than shipped; a lead departure reaches +0.5 one frame later and +1.0 at the same time). **Field test pending.** |
| D17 | Latched point follows a drifting endpoint | the forward extension (`EXTEND_RATE`/`EXTEND_DEADBAND`) needs only the model still calling the stop with latch confidence, not the latch window: route 25 t=1547 (field test 3) drifted 3 m beyond a frozen commitment and, with the 5 m setback, headed for a stop ~10 m short of the line |
| D16 | Arbitration with e2e | unchanged: `min()` over MPC, cruise, e2e and the committed profile; with a front-loaded profile the car is slower when the model's late demand would come, so e2e loses the early phase and is milder late. Excluding e2e while committed stays an option if a drive shows otherwise |
| D12 | Fallback to "stop as MPC obstacle only, no mode switching" | if after two fix rounds the phase-3 field test still shows a resume pulse or `shouldStop` dither at a real stop, or the driver had to break a hold at a green more than once per ~10 stops |

### Combo-only decisions

The code these rows record ships only on `combo`: the fork's opendbc and combo's green-light launch assist. They are numbered
apart from BLoTv3's rows so that a merge from `BLoTv3` never reuses a number. `5b27a56d1` introduced them as D32 and D33, which
BLoTv3's step 1 now uses.

| # | Decision | Ruling | Verification |
|---|---|---|---|
| C1 | The standstill-exit jerk limit, restored (2026-09-06) | SCC14 `JerkUpperLimit` gates when the car's cruise module commits to a standstill exit; it is a permission, not a command, and the realized acceleration ramp is ~140 ms whatever it says. The Palisade bracket (start-from-stop maneuver, one variable, command to wheel roll) was 1.0 = 1390 ms, 3.0 = 960, 5.0 = 790, 7.0 = 1200, and 5.0 was locked. It rode on the `starting` `LongCtrlState`; upstream deleted that state in July 2026 and the fork inherited the deletion through a sync merge, putting every launch back on the 3.0 arm with the 5.0 branch left as unreachable code. `acc_jerk_upper()` in the opendbc fork writes the old window out: the pid state while the wheels are still. The acceleration request is untouched, so nothing gains authority | `TestHyundaiLaunchJerk` in the opendbc fork (`opendbc/car/hyundai/tests/test_hyundai.py`); field test pending |
| C2 | A reset clears the whole launch state (2026-09-06) | the reset block cleared `anticipating_prev` but not `anticipating`, `launch_armed` or the `launch_open` filter, so re-engaging at a light with an open path read as a fresh green-light opening on the next frame and fired the launch edge that releases the landing law | `openpilot/selfdrive/controls/tests/test_longitudinal_planner.py` (`TestResetClearsTheLaunchState`); field test pending |

**Open on `combo` (2026-09-25, the owner's call):** two of BLoTv3's cases fail on `combo`'s LongControl and pass on BLoTv3's:
`test_longitudinal.py::TestPlant::test_the_stop_bit_hands_the_car_to_longcontrols_stopping_ramp` and
`test_longitudinal_planner.py::TestStopBit::test_a_target_nearer_than_the_kiss_can_stop_short_of_takes_the_stop_bit_at_once`.
Both expect stock LongControl's stopping ramp once the stop bit rises above `STANDSTILL_SPEED`; `smooth-stops`' hold waits for
0.10 m/s and its settle follows the plan until then (D37 (2)). Retiring the settle's speed wait (`want_hold`'s speed test,
`STOP_KISS_DECEL`, `SETTLE_JERK`) or accepting the corner on `combo` is open.

## 3. Module contracts

### longitudinal_lead.py
`LeadObservation.from_radar(lead, service_valid)` (filtered speed/accel, finite, `dRel > 0`),
`lead_present(radar_state)`, `relevant_lead(radar_state, v_ego, path_end)` (BLoTv2's distance/time
relevance rule — the only filtered presence check in the tree), `anchor_model_lead(model_lead, radar_lead)`
(BLoTv2's validity gate plus first-horizon acceleration/speed, computed once per frame; since the
2026-08-29 field test the gate tolerates `MODEL_LEAD_STATIONARY_NOISE` = 0.2 m/s of below-zero sensor
noise on a stopped lead — the strict `>= 0` gate dropped the anchor for 0.1–0.5 s chunks through every
lead launch, collapsing the departure forecast, re-raising the stop bit and resetting the hold release;
a reversing lead still fails closed), and the
closing-speed / `total_decel_requirement` / TTC physics. `total_decel_requirement` is
`max(closing_requirement, stop_requirement)`, not the sum BLoTv2's doc stated.

### necessity_supervisor.py
`NecessitySupervisor.update(lead, v_ego, a_mpc, predicted_lead_accel=None) -> LongitudinalPolicy(jerk_scale, t_follow_pad)`;
the follow time itself is assembled in `long_mpc.py` (`get_T_FOLLOW(personality) + t_follow_pad`).
Triggers, thresholds, slews, the whiplash ratchet (kept as its own guard) and both pad ceilings
(0.45 s onset braking, 0.75 s near-stopped lead) are BLoTv2's. Fixes: both pads hold at their ceiling
instead of vanishing — BLoTv2 gated them off above `ONSET_MAX_A_REQ`; the onset pad ramps to
`ONSET_FULL_DECEL` (1.5 m/s² of lead braking) and the near-stopped-lead pad to `STOPPED_LEAD_FULL_DECEL`
(1.2 m/s² of required deceleration, BLoTv2's field value), and `ONSET_MAX_A_REQ` is only the stand-down
gate; the low-speed hold latches only when the supervisor was necessity-braking in the frame before `v_ego`
crossed `MIN_SPEED`, and the emergency / lead-loss release paths clear it. A stand-down (low TTC, high need, a real
shortfall against the MPC's own braking) is internal: it returns the stock policy for that frame and reaches no alert and
no other module. `JERK_SCALE_MIN` is the single clip source for `long_mpc.set_weights`.

### long_mpc.py
Stock helper functions untouched; `update()` and `run(iterations)` are BLoTv3's. Per frame: `set_cur_state(v, a, a_told)` — `v`
the car's measured speed (D37), `a` the planner's seed (D33: the MPC's own plan, clipped, while it won the previous frame, else the
published target), `a_told` the target the planner published last frame (D32) — then one
`update(radarstate, personality, lead0_anchor, lead1_anchor, stop_x, jerk_scale, t_follow_pad, prev_accel_constraint)`,
which in order: builds the lead0/lead1/stop obstacle columns — the QP plans against the obstacles present this frame; finds the
binding column (`argmin(x_obstacles[0])`), whose source is `self.source` — the MPC's own, which the planner no longer overwrites;
names the binding obstacle's identity (`binding_obstacle`: `fake` for the fake lead's free run, `lead` for a real radar lead in
either slot, `stop` for the column) and decides a hand-over by D32's rules — the column's commit and release at once, a radar lead
taking over from the free run after `LEAD_PRESENCE_FRAMES` (2), a lead appearing ahead of the committed column at once, a lead that
has gone after `LEAD_ABSENCE_FRAMES` (7); a lead and the column trading the binding while both are present, and a real lead → real
lead change, are not hand-overs. The two counts time only the hand-over: through a dropout the QP already plans against the free run
from the first absent frame, and a lead back within six frames carries on without one. On a hand-over it sets `x0[2]` to `a_told`,
so the new owner's plan starts from what the car was told, refills `a_prev` with it (D14), restarts the iterate from the
constant-acceleration rollout of `x0` (`reseed()`: v = max(v0 + a0·t, 0), at rest once stopped, every u = 0) and solves that frame
with two SQP-RTI iterations; every other frame solves with one, continuing from the seed. Then it applies `jerk_scale`/`t_follow_pad`
only while lead0 is the source; calls `set_weights` exactly once; solves; scores `crash_cnt` against the lead trajectory the solve used
(model-anchored when valid, radar extrapolation otherwise — a disclosed departure from stock FCW
sensitivity; scoring a radar-only path against a model-anchored solve would produce phantom
warnings). After the solve, `a_next` = `interp(dt, T_IDXS, a_solution)` is the plan one frame on — exact, since the
acceleration is linear between nodes, and unclipped: the plan, not what the car was told; the planner clips it for its seed.
`reset()` zeroes `a_told` beside `x0`. The third-lead machinery is removed. The MPC's acceleration bound stays opendbc's
`ACCEL_MAX` as in stock (BLoTv2's `min(ACCEL_MAX, 4.0)` always equalled it); only the cruise
envelope carries the 4.0 m/s² launch request.

### stop_landing.py
`StopLanding(dt, action_t, release_jerk)`, with `action_t` the planner's `longitudinalActuatorDelay + DT_MDL` and `release_jerk` the
planner's own cruise jerk schedule (`J_CRUISE_BP`, `J_CRUISE_VALS`), passed in, not imported (the planner imports this module).
`StopLanding.update(a_target, v_ego, lead, stop_intent, launch=False, a_ego=None) -> a_target` keeps the arbitrated target inside the
landing corridor (D22, D34) while a landing is live: intent with a plan braking more than `KISS_DECEL` below `LANDING_SPEED` starts it,
unless the planner is issuing a launch on that same frame. A landing ends on a launch — a corroborated lead departure or a Force Stops
release, the frame the commitment is over (the result carries no stop point) — on the raw plan positive for `LAUNCH_FRAMES`
frames while rolling above `KISS_SPEED`, or on the speed reaching `LANDING_SPEED`. The hold bit also drops when the car rolls past `RESUME_SPEED` and the hold
becomes a moving commitment again; that frame is not a launch and the landing survives it. On every exit after a frame where the
landing held the output below the plan it releases (`releasing`, D38): the output is `min(what the law publishes, last output +
J(v)·dt)`, J from `release_jerk` at the current speed, until the law's own result is at or below that ceiling; braking passes at
once, a landing that re-latches meanwhile keeps the release, and an exit from a frame where the landing passed or lifted the plan
carries nothing. `output` is the last output. A non-finite `a_target` passes through untouched, leaving the latch, the integral and a
release alone; a non-finite `v_ego` resets everything, a release included. `landing_bound(v)` / `landing_floor(v)` are the corridor as the
car's deceleration against its speed; both are `KISS_DECEL` at and below `KISS_SPEED`. The request edges `bound(v)` /
`floor(v)` are those tables with every breakpoint below `LANDING_SPEED` moved up to the speed from which a car riding that
edge reaches it after `action_t` (`request_breakpoints`; at 0.55 s bound 0.531 / 1.524 m/s, floor 0.493 / 1.178 m/s); the
3.5 m/s ceiling stays. The planner computes intent as: a committed stop or hold, the MPC's own horizon ending below
`STOP_INTENT_SPEED` (a stopped lead, the committed column), or the model calling a stop in Experimental mode
(`should_stop`/`strict_stop`), and sets the plan's stop bit around the landing (D37: withheld while the kiss lands a rolling car; the
planner reads `lead_requirement()` and `bound()` for that). The lead is the planner's `LeadObservation` — the raw radar lead, with no
absence hold, so a one-frame dropout of a close lead lifts the output for that frame; `lead_requirement(v, lead)` is the larger of
`longitudinal_lead.total_decel_requirement` for a stop
`STOP_DISTANCE` behind the lead in at least `LEAD_MIN_GAP_BUDGET` of room, the braking that stops the car
`LEAD_MIN_GAP_BUDGET` short of the lead in the room left after `action_t` (unbounded as that room closes), and, for a braking
lead, the same against where the lead will stop. The output is `max(min(plan, −floor), −bound, ACCEL_MIN)` with
`floor = floor(v) + press` (applied only where it is above 0) and `bound = max(bound(v) + press, lead_requirement)`: no
close-lead switch, and no loop on the measured acceleration in either edge. `a_ego` (the car's measured acceleration) feeds
only the anti-stall integral (D34): while `v > STANDSTILL_SPEED` (0.10 m/s) it integrates `a_ego + landing_floor(v) −
STALL_DEADBAND` at `STALL_GAIN`, clipped every frame to `[0, STALL_MAX]` fading to 0 between `CREEP_SPEED` and
`CREEP_FADE_SPEED`; the press is that integral, held and not applied at standstill; it resets when a landing starts. `reset()`
on the planner's reset (disengage, gas) drops the latch, the integral and a release.
On `combo`, LongControl's thin handoff (`smooth-stops`) owns the clamp deferral and its own kiss below the stop bit; since D37
withholds the bit while the kiss lands a rolling car, LongControl stays in pid through the landing and the settle no longer runs
there (retiring it is the owner's call). Ruling (audit 2026-09-02): every candidate is subject to the corridor, the curve
candidate included — the corridor only removes surplus braking, the reaction brake regime (≥ 3 m/s) is never inside its bound,
and exempting a candidate would reopen the harder-than-the-law landing class.

### longitudinal_planner.py
Envelope (`a_max = 0.6 + 3.4 (1 − v/40)³`, clamped by opendbc's `ACCEL_MAX`), jerk schedule, ordinary-cruise
comfort shaping (`ordinary_cruise_comfort_enabled(experimental_mode, force_decel, radar_valid)`) as BLoTv2. No lateral
turn budget (D2). `update()` order: reset state (the output, cruise and the next MPC seed start from the car's `aEgo`; the
supervisor, the pre-release and the landing reset) → lead and anchors → supervisor → stop observation →
`force_stops.update(..., model_valid, output_a_target)` → `mpc.set_cur_state(max(v_ego, 0), a_seed, output_a_target)`
→ `mpc.update(...)` → `fcw = mpc.crash_cnt > 2 and not standstill` → the launch edge → candidate membership and the cruise
carrier → candidates → `min()` → the landing law → the stop bit → the clip. `output_a_target` in both calls is the previous
frame's published target, read before this frame writes it.
- Speed (D37): the MPC plans from `vEgo`; there is no `v_desired_filter`, so `longitudinalPlan.speeds[0]` is `vEgo`. `init_v`
  stays in the constructor's signature, unused, because upstream's plant passes it.
- Seed (D33): `a_seed = clip(mpc.a_next, ACCEL_MIN, ACCEL_MAX)` when the previous frame's winner was the MPC (`mpc` or `column`) —
  also while the landing bounds or hands back what is published from it — else the published target; D14's refill and a hand-over
  read the published target as `a_told` (D32).
- Candidates carry an identity: the MPC (`column` while its stop column binds, else `mpc`), `cruise`, `e2e` (Experimental
  mode with a valid model, and against a moving profile only when more urgent by `E2E_STOP_MARGIN`, D20 — with no incumbency, so
  a profile joining from a target e2e set can evict it: D37's open item) and `profile` (the committed approach profile). `min()`
  gives the target, the published `plan_source` and the kept `plan_winner`; `mpc.source` is not overwritten.
- Bumpless transfer (D33): when the previous winner is not in this frame's candidate set, cruise slews from the published target
  instead of its own value, so it carries the hand-over at `J_CRUISE(v)`; on every other frame it slews on its own value. What the
  landing held is not the planner's to carry: the landing hands it back itself (D38). Cruise's stop bit is
  `should_stop(v, v_cruise − v_ego)`, from the set speed.
- The landing law's launch edge is `lead_departing or (holding_prev and not holding and stop_x is None)`: only the end of the
  commitment counts, never a creep resume that turns a hold back into a moving commitment. `StopLanding` is built with the
  `action_t` the plan is read at and the planner's `(J_CRUISE_BP, J_CRUISE_VALS)`.
- Stop bit (D37): while the landing is latched, the bit has not yet risen in it, `v_ego > STANDSTILL_SPEED` and the landing's
  `lead_requirement(v, lead)` is within its `bound(v)`, the kiss owns the stop and the bit is `force_stops.holding` alone;
  otherwise it is `force_stops.holding or any(candidate stops) or (landing and should_stop(v, landed target))`. Once the bit has
  risen inside a landing it stays for as long as that rule holds it (`landing_stop_bit`), until the landing ends.

### stop_helpers.py
`observe_model_stop(model, car_state, radar_state) -> StopObservation` — BLoTv2's tiers
(`shouldStop`, strict trajectory, early high-speed, early hint; BLoTv2's missing-velocity fallback
tier cannot occur with complete typed messages and is gone), straight-approach guard, relevant lead,
committed turn, and per frame the launch-evidence and corridor verdicts. `stop_release_open(model)` — one definition, non-braking not
required (combo's field-tested semantics). `leads_clear_of_stop_path(model, path_end)` — fails
closed unless **every** model-lead hypothesis with probability > 0 is outside the corridor, and on
any shape/finite irregularity (route-29 negative sentinel); a flat path, which is what the model
publishes at standstill, is a legal straight corridor, a reversing one is not. `MODEL_INVALID_RELEASE_S = 0.5` is
defined here and shared. Typed capnp access; no `getattr` guards.

### force_stops.py
`ForceStops.update(obs, CS, experimental_mode, enabled, model_valid, a_prev_output) -> (v_cruise_cap, stop_x, holding, a_target)`;
the observation carries lead presence/relevance, launch evidence and the corridor verdict, `enabled` is the planner's own
active signal and `a_prev_output` the target the planner published last frame (a non-finite one reads 0). Force Stops reads no
`aEgo`.
States: `idle → shaping → committed → holding → (committed | idle)`.
- Entry requires Experimental mode (**entry only** — a later mode exit never releases a hold), no
  raw lead, a valid model, BLoTv2's tiers, path-length window and latch confidence, plus
  the committed-turn veto. Pre-latch shaping cap on the live endpoint as BLoTv2.
- `committed`: `remaining` decremented by ego travel, forward-ratcheted toward a re-extending
  endpoint (`EXTEND_RATE`/`EXTEND_DEADBAND`) and down-ratcheted toward a collapsing one
  (`DOWN_RATE`/`DOWN_DEADBAND`), each after `FOLLOW_CONFIRM_S` of evidence past the deadband (D30);
  below `DOWN_SPEED` the follow-down is immediate (route 38 t=351); `LATCH_SETBACK`; `stop_x = max(remaining, −STOP_DISTANCE)`;
  no cruise cap (`NO_CAP`, D19); `a_target` = the committed approach profile (D13), entered on
  its first frame from `min(a_prev_output, 0)` and slewed at `PROFILE_JERK` on its own value from there (D37), a plan
  candidate with source `stop` while the commitment is moving. Model invalid releases only after
  `MODEL_INVALID_RELEASE_S`. A tracked lead (not one radar frame) hands a moving commitment to the
  lead logic (D15). A non-positive endpoint (`path_end ≤ 0`: at walking pace the model's stopped plan ends a few
  centimetres behind the car) is the stop, not a release: it resets only an idle or shaping detector, and only after the
  standstill hold decision; a commitment runs through it at any speed, reading it like any endpoint inside the setback —
  qualify, the windows, the latch and the green all need a positive endpoint — its point dead-reckoning within
  `DOWN_DEADBAND` of the car and followed in at `DOWN_RATE` beyond, above `DOWN_SPEED` only after `FOLLOW_CONFIRM_S` (D35).
- `holding`: entered at `CS.standstill` while committed, or within 10 s of a lead or a gas tap breaking
  a commitment or a hold when the car is stopped with stop evidence (on a non-positive endpoint that is the model's stop
  call, D35); `stop_x` is the committed point clipped to at most 0 — dead-reckoned when the car stopped past it, never below
  −`STOP_DISTANCE` — and `holding` forces `shouldStop`, so `controlsd`'s `cruiseControl.resume` cannot pulse. Leaves to `committed` (not
  idle) at `v_ego ≥ 0.8 m/s`, so an unsigned wheel-speed flicker on a grade never drops the latch. That resume is not a
  release: the commitment carries on with the latch, and it starts its release and follow evidence over — the hold's own
  counts and the evidence from before the stop are not this commitment's.
- Release to idle: a green — a path longer than `RELEASE_OPEN_LENGTH` (`PATH_OPEN_LENGTH`, 30 m, the hold's standstill
  calibration) for `RELEASE_OPEN_FRAMES` frames with no stop evidence (`shouldStop` or strict), one test applied identically
  to a hold and to a moving commitment (a long path the model still calls a stop on is not a green). A moving commitment
  forms with the path end 40–65 m out, so at approach speed the length is always met and only the stop evidence gates the
  release (D28's cross-reference); F011's plan-physics green was built and held back (D35). Also: filtered launch evidence
  (`stop_release_open`, 0.30 s time constant); gas; brake; a **relevant** lead; model invalid ≥ 0.5 s; the D10 fallback; the
  slow release (the detector below `RELEASE_THRESHOLD` with the 4 s position hold spent); a lane change (D21). Every release ends the
  commitment whole — the approach profile's anchor and the open-path count go with it, so the next commitment enters its
  ramp from the published target (D37) and its green release from zero. Fast re-entry: a lead **or** a gas tap that
  breaks a hold **or** a moving commitment arms `REARM_S`; inside that window, if the car is at standstill again with stop
  evidence present, `holding` is re-entered directly; the 10 s gas grace suppresses only the shaping cap.
- Non-goal: no committed-lifetime + delay-projection scheme to move `shouldStop` earlier — tried
  on route 29 in BLoTv2 (0.450 s landed inside the 0.5 s actuator delay and weakened a fail-closed
  release). Holding engages at standstill only.

### Maneuver plant (`plant.py`, `maneuver.py`, `test_longitudinal.py`)
`Plant(..., actuator_lag=False, CP=None, accel_error=0.0, creep=0.0)`: the planner, a stock `LongControl(CP)` and the car interface are built on `CP` —
the Civic's by default, for upstream's maneuvers, and `palisade_car_params()` (openpilot longitudinal,
`longitudinalActuatorDelay` 0.5, so `action_t` 0.55 s) for BLoTv3's stop, landing, red-light and lead maneuvers (D36).
Each model frame runs five control ticks: LongControl on the held target and stop bit (the stop bit reaches its stopping
state — this is BLoTv3's stock LongControl, not `combo`'s settle), then the actuator — ideal, or `ActuatorLag`, the
CAN-identified per-band dead time, bite and release time constants and gain, which held wheels do not wind — the wheels,
and the car's own speed filter, which gives `carState.aEgo`; `vEgo` is the wheel speed. `accel_error` is added to the car's
acceleration while it rolls, whatever it was asked (brakes that fall short, and the same push at cruise and on a launch; held
wheels do not feel it); `creep` is driveline creep, `creep·(1 − v/DRIVELINE_CREEP_SPEED)` (1 m/s), added whenever the request is
milder than `CREEP_HOLD_REQUEST` (−0.2) — an assumed model, not an identified one; both are inert at 0. Radar `dRel` is published on
`RADAR_DREL_STEP` (0.1 m, the radar DBC's `LONG_DIST` factor), and the fake red-light model raises `shouldStop` as modeld
does. `step()` also returns `a_target` (the plan) and `forcing`, which the maneuver log carries as columns 7 and 8; the
forceDecel check is upstream's, judged on the plan. Every maneuver class that runs the planner is parametrised over
`ACCEL_MAX_PINS` (2.0, 4.0), patching opendbc's, the planner's and the MPC's `ACCEL_MAX` in `setUp`.
The continuity gate (`TestRedLightPlanContinuity`): red lights at 14/17/20 m/s with the line 11.5·v out, through the
actuator; the window runs from the first `aTarget < −0.3` (the fake model's call) to the last frame planned above 1 m/s; it
asserts the stop inside [line − 5 m, line], Force Stops committing strictly after the call and inside the window, no
one-frame step above `PLAN_JERK_MAX`·`DT_MDL` (0.15), and no turning point of `PLAN_REVERSAL` (0.15) before the final ease.
At 22 m/s (`TestRedLightPlanContinuityAtTheColumnsNeed`) that assert is a strict expected failure — the stop column's converged
need at the car's true speed steps 0.208 on the commit frame — and the companion `test_only_the_commit_steps_beyond_the_bound`
lets only that frame step, braking by no more than the need (0.206) plus the MPC's 0.1 first-solve tolerance. On an ideal
actuator the red lights at 14–22 m/s are plain tests (D37's entry). `TestACarThatBrakesLessThanAsked`: 10 m/s / 90 m and
15 m/s / 120 m toward a stopped lead with `accel_error = BRAKE_SHORTFALL` (0.3): no contact, a stop, never nearer than
`LEAD_CLOSEST_GAP` (2 m), the continuity rule above 1 m/s; resting `LEAD_REST_GAP` (4 m) back is a strict expected failure
(2.94–2.98 m, step 3). `TestALeadLeavingALatchedLanding`: 10 m/s toward a stopped lead at 70 m; once the landing is latched —
rolling at 1.0 m/s without creep, 1 s at rest without creep, or rolling at 0.6 m/s with `DRIVELINE_CREEP` (0.5) — the lead
inches forward or pulls away, and every frame that starts with the target held below the plan the planner gave the landing
rises at most `J_CRUISE(v)`·dt, judged without reading the law's own flags (D38). Expected failures are
`unittest.expectedFailure`, strict under pytest and CI's unittest runner alike, with the cause in the comment at the marker.

### conditional_experimental_mode.py
Runs every control tick (watchdogs, pedals, timers at `DT_CTRL`); evidence acquisition only on new
model frames. Entry filter, debounce, hysteresis, the 3 s recent-lead guard with corridor release,
committed-turn veto and post-stop/override suppression as BLoTv2. Fix: on control ticks a raw lead
may only revoke a *pending* recent-lead release, never wipe entry evidence; entry vetoes use
`relevant_lead`. Model and radar validity are separate inputs: completeness — and with it the `MODEL_INVALID_RELEASE_S`
release — is judged on the model frame alone, while an invalid radar only empties that frame's stop evidence (every tier is
read against the radar leads) and revokes a pending lead release. Exits: resumed motion, stable clear, pedals, invalid model
(radar validity is not part of that judgment). selfdrived hook:
`experimental_mode = openpilotLong and (manual or conditional)`; pedal (driver-override) suppression lives inside
`ConditionalExperimentalMode.update()`, not in the hook.

## 4. Behavior: preserved vs changed
Preserved: envelope, comfort, jerk schedule, personalities (aggressive 1.0 s), supervisor triggers,
model-lead anchoring (lead0/lead1), lead-departure pre-release, CEM tiers/timers, Force Stops
shaping/commit/ratchets/setback/`DV_MAX`, `STOP_DISTANCE` 7 m, acceleration-change cost through
standstill, e2e candidate dropped while the model is invalid.
Changed: CEM can enter with an irrelevant lead in view; hold through standstill owned by Force
Stops and not released by a mode exit; FCW in stock's form, scored against the solved trajectory;
manual Experimental survives pedal taps; pad saturation and low-speed hold; third lead removed;
no turn budget; no cross-process stop fields; `longcontrol.py` is stock.

## 5. Phases and gates
Every phase: behavioral unit tests (real capnp messages, no constant echoes), the longitudinal
maneuver suite with an honest liveness shim, an rlog first-divergence replay against BLoTv2, then
the owner's field test before the next phase.
0. Branch, this document, harness (`plant.py` shim with real `valid/alive/freq_ok`, `enabled`,
   fully populated `leadsV3`, maneuvers that drop `radarState`/`modelV2`), longitudinal replay
   tooling with a pinned route manifest (d7, 29, d2, d9, 17, 27), and an audit of the combo-only
   Force Stops/CEM reconciliation commits.
1. Cruise layer. Field: Chill, no lead — urban and highway set-speed steps, one on-ramp; the D4
   launch metric.
2. Lead layer. Field: following, stopped-lead approach, pull-away, queue creep, a moderate lead
   brake from ~40 m (crosses 1.5 m/s²), a late low-speed lead brake (hold at partial softening).
3. Stop layer (also re-touches `long_mpc.py`/`longitudinal_planner.py` to wire `stop_x`/`holding`;
   the phase-2 lead-only replay is re-run with Force Stops idle first). Field: red lights with and
   without distant traffic, green release from hold, gas tap while holding then re-stop, signaled
   low-speed turn, a stop on a grade; D12 checkpoint.
4. Integration into `combo`, staged and each field-gated: (a) BLoTv3 alone on a scratch
   integration branch; (b) `smooth-stops` re-authored against BLoTv3's `LeadObservation` and its
   radard fix landed with a test, the dead BLoTv2 `longcontrol.py` clamp dropped; (c)
   `curve-speed-limit` composition; (d) SOL/AOL `selfdrived.py` merge (D11). Then retire branches
   and drop the audited combo-only commits.

## 6. Verification map
Envelope samples on the requested curve; comfort;
arbitration incl. hold; no FCW from the removed stand-down path; each following-time pad at its ceiling past the
stand-down gate (the onset pad's ramp ends at 1.5 m/s², the near-stopped-lead pad's at 1.2);
low-speed hold at partial softening with emergency/lead-loss release still reaching 1.0; whiplash
ratchet and hold in one scenario; row-0 policy; single `set_weights`; `a_prev` refill on the exact
handoff frame; corridor rule with a 0.2-probability hypothesis; commit → hold → every release path;
flickering model stop signal while holding never drops `shouldStop`; grade flicker returns to
`committed`; lead passes through then fast re-entry; gas tap re-stop; CEM entry with a far lead;
model hang releases within 0.5 s; pedal latency; the selfdrived hook keeps manual mode under
override. Landing law: the corridor's shape and its request edges one actuator delay ahead, window, intent latch through a flicker, a close lead's requirement continuous in distance and never blocked, the anti-stall integral, creep floor; the planner bounds whichever candidate lands (e2e at walking pace), not above the window, and passes what a close lead needs; plant: a stopped lead, a red light and a hard close lead stop all land inside the law and still stop, and the model's late ramp (`e2e_landing_push`) is bounded with the law and not without.

Added by the 2026-09-17 audit pass, by name (paths under `openpilot/selfdrive/controls/tests/` unless noted):
- `test_controlsd_long_state.py::test_the_published_state_is_the_one_the_engage_frame_computed`,
  `::test_the_published_state_enters_stopping_on_the_frame_that_asks_to_stop`,
  `::test_the_published_state_leaves_stopping_on_the_frame_that_releases_the_stop` — D31: the published
  `actuators.longControlState` is the state the same frame's `LoC.update()` computed, on the engage, stop and release frames.
- `test_force_stops.py::TestMovingReleases::test_the_slow_release_ends_the_profile_with_the_commitment`,
  `::TestFollowConfirmation::test_a_new_commitment_starts_with_no_follow_evidence`,
  `::TestHold::test_a_gas_tap_that_breaks_a_moving_commitment_arms_the_hold_re_entry`,
  `::TestHold::test_a_long_path_the_model_still_calls_a_stop_on_does_not_release_the_hold`,
  `::TestHold::test_a_hold_that_rolls_again_counts_its_release_and_follow_evidence_over`.
- `test_conditional_experimental_mode.py::TestExitAndHold::test_a_nonfinite_trajectory_still_releases_the_mode_while_the_radar_is_down`
  and `::TestExitAndHold::test_a_radar_dropout_alone_is_not_an_invalid_model`.
- `test_longitudinal_planner.py::TestHoldRelease::test_a_creep_resume_from_a_hold_is_not_a_launch_and_the_landing_survives`
  and `::TestHoldRelease::test_a_hold_released_by_an_open_road_still_ends_the_landing`.
- `test_stop_landing.py::TestLatchAndLaunch::test_a_nonfinite_target_leaves_the_plan_alone_and_does_not_corrupt_the_launch_count`.
- `test_necessity_supervisor.py::TestPads::test_the_stopped_lead_pad_tops_out_below_the_stand_down_gate` — the
  near-stopped-lead pad is at its ceiling between `STOPPED_LEAD_FULL_DECEL` and `ONSET_MAX_A_REQ`.
- `openpilot/selfdrive/test/longitudinal_maneuvers/test_longitudinal.py::TestEnsureStartLaunchScope::test_ensure_start_ignores_gap_settling_once_above_the_launch_band`
  — the harness's `ensure_start` check is scoped below 2 m/s (ported from `combo`), so gap settling above the launch band
  no longer fails a maneuver.
- `test_long_mpc.py::TestUpdateProtocol::test_committed_stop_is_a_fixed_obstacle` now pins `STOP_DISTANCE` at 7.0 (D3).

Added by the 2026-09-25 audit step 1, first build (D32–D36), by name (paths under `openpilot/selfdrive/controls/tests/` unless noted):
- `test_long_mpc.py::TestNewBindingObstacle::test_a_commit_after_a_free_run_starts_at_the_converged_plan` (7 commits × both
  personalities × both pins: within 0.1 of converged, a refill, more than one iteration on the commit frame),
  `::test_the_first_solve_of_a_new_obstacle_does_not_depend_on_the_previous_one`,
  `::test_a_real_lead_after_a_free_run_is_a_new_obstacle`,
  `::test_a_stopped_lead_appearing_after_a_free_run_is_braked_for_from_its_second_frame` (renamed in round 2),
  `::test_a_stopped_lead_appearing_before_the_stop_point_is_braked_for_at_once`,
  `::test_a_commit_behind_a_lead_is_a_new_obstacle`,
  `::test_a_lead_that_drops_out_is_the_same_obstacle_until_it_has_left`,
  `::test_a_column_and_a_stopped_lead_in_the_same_place_are_one_obstacle`,
  `::test_the_same_car_changing_slots_is_the_same_obstacle` — D32; `TestUpdateProtocol::test_a_next_is_the_plan_one_frame_on`;
  `TestUpdateProtocol::test_handoff_from_an_adaptive_lead0_reanchors_the_change_cost` is now
  `::test_handoff_from_an_adaptive_lead0_to_the_stop_column_reanchors_the_change_cost` (a lead0 → lead1 swap of one car is no
  longer a hand-over).
- `test_longitudinal_planner.py::TestHandOver::test_a_mode_exit_hands_over_from_the_output`,
  `::test_a_green_that_releases_a_committed_stop_hands_over_from_the_output`,
  `::test_the_profile_ending_while_the_column_stays_hands_over_from_the_output`,
  `::test_a_lead_leaving_a_standstill_ends_the_kiss_and_the_launch_starts_from_it`, and the guard
  `::test_a_lead_pulling_away_is_released_at_the_mpc_s_own_rate` (fails if cruise is anchored every frame);
  `TestMpcSeed::test_an_edit_of_the_published_target_does_not_ring_through_the_mpc`,
  `::test_an_edit_of_the_published_target_does_not_ring_through_the_committed_column`,
  `::test_the_mpc_starts_from_the_published_target_when_another_candidate_drove`,
  `::test_a_driver_override_restarts_the_mpc_from_the_car`, `::test_the_seed_gives_nothing_back_at_a_braking_onset`,
  `::test_at_standstill_the_mpc_does_not_mirror_the_kiss` — D33. Rewritten:
  `TestHoldRelease::test_a_hold_released_by_an_open_road_still_ends_the_landing` is now
  `::test_a_hold_released_by_an_open_road_ends_the_landing_and_launches_from_the_kiss` (the output leaves the kiss at
  `J_CRUISE`·dt), and `TestStopLanding::test_the_law_is_off_above_its_window_and_next_to_a_close_lead` is now
  `::test_the_law_is_off_above_its_window_and_lets_a_close_lead_through` (the requirement passes; the 5 m exemption is gone).
- `test_stop_landing.py::TestCorridor::test_each_edge_is_requested_one_actuator_delay_ahead_of_the_car` (recomputes the
  breakpoints by forward-integrating a car riding each edge), `::test_without_a_delay_the_request_is_the_car_side_corridor`,
  `::test_the_bound_only_removes_braking_and_the_floor_only_adds_it_up_to_itself`,
  `::test_a_plan_inside_the_corridor_passes_untouched`;
  `TestNoLoopOnTheMeasuredDeceleration::test_the_request_does_not_depend_on_a_car_braking_beyond_the_floor`;
  `TestLead::test_the_output_is_continuous_as_a_close_lead_crosses_5_m`,
  `::test_the_output_is_continuous_as_a_stopped_lead_closes`,
  `::test_the_braking_passed_grows_without_limit_as_the_room_left_after_the_delay_closes`,
  `::test_a_lead_inside_the_gap_budget_gets_what_stops_the_car_short_of_it_after_the_delay`,
  `::test_a_plan_braking_less_than_the_leads_requirement_is_left_alone`;
  `TestAntiStall::test_a_car_not_slowing_presses_both_edges_more_every_frame_up_to_the_cap`,
  `::test_the_press_drains_as_the_car_slows_again`,
  `::test_nothing_is_pressed_at_standstill_and_the_press_is_held_for_the_next_roll`,
  `::test_nothing_winds_it_up_where_the_floor_has_faded_out`, `::test_a_press_fades_in_and_out_with_the_floor`,
  `::test_a_car_creeping_under_a_hovering_plan_is_pressed` — D34. Deleted with their mechanisms: `TestReleaseLift`,
  `TestCreepPress`, `TestWatchdog` and `TestLead::test_a_close_lead_lifts_the_bound_but_keeps_the_floor`.
- `test_force_stops.py::TestStopHere::test_a_commitment_keeps_its_point_through_a_non_positive_endpoint_and_forms_the_hold`
  (with and without `shouldStop`),
  `::test_a_point_beyond_the_deadband_follows_it_in_like_any_endpoint_inside_the_setback`,
  `::test_a_broken_commitment_re_enters_the_hold_on_a_non_positive_endpoint` (gas, lead),
  `::test_a_non_positive_endpoint_still_ends_shaping` — D35.
- `openpilot/selfdrive/test/longitudinal_maneuvers/test_longitudinal.py::TestRedLightPlanContinuity::test_the_plan_builds_to_the_stop_without_steps_or_let_offs`
  (since round 2 14/17/20 m/s × both pins; 22 m/s moved below), `TestRedLightPlanContinuityOnAnIdealActuator::test_the_plan_builds_to_the_stop_without_steps_or_let_offs`
  (strict expected failures at 14/17/20 m/s on the first build; since round 2 plain tests at 14–22 m/s × both pins, D37),
  `TestPlant::test_the_radar_reports_the_lead_in_its_own_distance_steps`,
  `::test_the_planner_sees_the_cars_speed_filter_not_the_applied_acceleration`,
  `::test_the_stop_bit_hands_the_car_to_longcontrols_stopping_ramp`, `::test_held_wheels_do_not_wind_up_the_actuator`,
  `TestTurningPoints::test_a_gradual_let_off_and_re_brake_counts_like_a_step` — D36. `TestLongitudinalControl`,
  `TestEnsureStartLaunchScope`, `TestRedLightStop` and `TestStopLanding` now run at both `ACCEL_MAX` pins, the last two on
  Palisade CarParams.

Added by the 2026-09-25 round 2 (D32's told-target hand-over, D33's clip, D35 at approach speed, D36–D38), by name:
- `test_long_mpc.py::TestNewBindingObstacle::test_a_lead_that_has_left_hands_over_from_what_the_car_was_told` (both pins ×
  aggressive/standard), `::test_a_stop_column_easing_off_hands_over_from_what_the_car_was_told` (to a stopped lead ahead of it and
  to the free run, both pins × both personalities), `::test_what_the_car_was_told_starts_a_new_obstacle_only` — D32. The two urgent
  hand-over tests now run both personalities, each bound tied to its measured worst (aggressive 0.11; standard 0.145 fake → lead,
  0.25 column → lead).
- `test_longitudinal_planner.py::TestHandOver::test_a_lead_that_leaves_hands_over_from_what_the_car_was_told` (both pins) — D32;
  `TestMpcSeed::test_a_plan_beyond_what_the_car_can_do_is_not_continued` (both pins) — D33's clip;
  `TestMpcSeed::test_the_mpc_plans_from_the_measured_speed` (the car answers through stock LongControl and the plant's actuator,
  nominal and `BRAKE_SHORTFALL` short, and must rest clear of the lead), `TestStopBit::test_the_kiss_lands_a_rolling_car_and_the_stop_bit_rises_at_standstill`,
  `::test_a_target_nearer_than_the_kiss_can_stop_short_of_takes_the_stop_bit_at_once`,
  `::test_a_speed_reading_after_the_stop_keeps_the_stop_bit_until_the_car_rolls_again`,
  `TestHoldRelease::test_a_hold_keeps_its_stop_bit_while_the_car_rolls`,
  `TestHandOver::test_a_commit_joins_from_the_published_target_not_the_measured_acceleration` — D37;
  `TestHoldRelease::test_a_green_launches_when_the_model_lets_go_a_frame_after_the_path_opens`, a strict expected failure: after a
  Force Stops green, a candidate still braking below the kiss on the next frame re-latches the landing at rest and nothing ends it
  (F035's launch gap, step 3; 0 of 77–82 field standstill exits).
- `test_stop_landing.py::TestRelease::test_a_launch_from_the_kiss_is_handed_back_at_the_cruise_carrys_rate`,
  `::test_a_launch_from_the_floor_while_rolling_is_handed_back_at_the_cruise_carrys_rate`,
  `::test_a_plan_climbing_out_of_a_pressed_landing_is_handed_back_at_the_cruise_carrys_rate`,
  `::test_the_press_adds_nothing_to_the_exit_step`, `::test_a_nonfinite_target_during_a_release_passes_and_the_release_carries_on`,
  `::test_only_what_the_landing_held_is_carried`, `::test_a_landing_starting_again_during_a_release_does_not_drop_it`,
  `::test_the_speed_leaving_the_window_does_not_cut_a_release_short`; `TestLatchAndLaunch::test_reset_forgets_the_landing` is now
  `::test_reset_forgets_the_landing_and_its_release` — D38.
- `test_force_stops.py::TestApproachProfile::test_the_profile_enters_from_the_published_target_not_the_measured_acceleration` (three
  entries), `::test_after_its_first_frame_the_profile_slews_on_itself_whatever_sets_the_output` — D37;
  `TestStopHere::test_at_approach_speed_a_non_positive_endpoint_pulls_the_point_in_only_after_the_confirmation` — D35.
- `openpilot/selfdrive/test/longitudinal_maneuvers/test_longitudinal.py::TestPlant::test_the_car_can_fall_short_of_its_request_and_creep_until_braked`,
  `TestRedLightPlanContinuityAtTheColumnsNeed::test_only_the_commit_steps_beyond_the_bound` and
  `::test_the_plan_builds_to_the_stop_without_steps_or_let_offs` (22 m/s × both pins, a strict expected failure),
  `TestACarThatBrakesLessThanAsked::test_it_stops_clear_of_a_stopped_lead_without_steps_or_let_offs` and
  `::test_it_rests_at_least_the_rest_gap_behind_it` (a strict expected failure; 10 and 15 m/s × both pins),
  `TestALeadLeavingALatchedLanding::test_the_landing_hands_back_what_it_held_at_the_cruise_jerk` (five cases × both pins) — D36–D38.
  `TestStopLanding::test_a_car_that_lets_go_of_the_brake_slowly_still_lands_close_to_the_law`, red on the first build, passes.

These unit tests run under pytest. CI's runner (`tools/test_runner.py`, a unittest loader) collects only `unittest.TestCase`
classes, so it runs none of the 312 pytest-style unit tests in step 1's files (`test_force_stops.py` 54, `test_long_mpc.py` 160,
`test_longitudinal_planner.py` 49, `test_stop_landing.py` 35, `test_stop_helpers.py` 14) — only the maneuver suite (59 passed and
6 strict expected failures at the end of step 1); this predates step 1 and is the same on `combo`. It is not fixed here: run the
controls, selfdrived and maneuver suites by hand, at both `ACCEL_MAX` pins, before any push.

## 8. Field test log

**2026-08-29, route 23 (combo 83ccd10ab5), Palisade.** Owner: "accelerating out of a curve feels held back"; "at
stops with a lead, when it was time to accelerate, it felt like a harsh jolt rather than a smooth and quick
switch"; no difference noticed otherwise.

- Curve: the lateral turn budget (D2) was the only thing clipping cruise acceleration in bends. Removed.
- Lead launch, from the rlog and the CAN bus (TCS13 / SCC12 / SCC14 decoded):
  1. The car's ESP runs a fixed ~1.3–1.5 s standstill-exit sequence after `StopReq` drops, ignoring the
     acceleration request, then snaps its own reference to ~0.4 m/s² *above* the request at ~11 m/s³. The lurch
     therefore scales with whatever the plan asks at that instant; `JerkUpperLimit` does not gate it.
  2. Every launch's request had already climbed to 1.3–1.8 m/s² by then because the hold released late:
     `anchor_model_lead` rejected the lead's forecast on the −0.00…−0.04 m/s a stationary lead reads on both
     sensors, so the anchor flapped through the launch, the departure forecast collapsed, the stop bit flickered,
     the pre-release cancelled and the hold-release grace restarted. Fixed with a 0.2 m/s stationary-noise
     tolerance (reversing still fails closed). Like-for-like replay of the four launches: sustained release
     0.45–0.60 s earlier on three, stop-bit flicker 3 → 1, 37 short whole-route differences, all smoother.
  3. Smooth Stops' extra 0.5 s hold-release grace for a stopped radar lead was pure launch latency on top of the
     car's own sequence; removed on `smooth-stops` (10-frame debounce for every stop, immediate on a measured
     departing lead).
- Not changed: no launch-staging cap in LongControl — with the hold releasing ~0.9 s earlier the car should break
  free while the plan is still 0.3–0.5 m/s². Reassess after the next drive.

**2026-08-29, route 24 (combo 2f7ba629d0).** Two disengagements at red lights ("didn't feel like we'd stop in time"),
both Experimental-mode e2e stops without a lead; a launch felt odd. Owner baselines from the same drives: comfortable
stops from 18 m/s are ≈ −1.5 m/s² held for ~10 s then eased to −0.6; comfortable launches peak at 4 m/s² and hold ~3
to 4 m/s, tapering to 1.2 by 12 m/s. The car's SCC saturates at ≈ 1.6–1.9 m/s² for any request above 1.5 (CAN
`ACCEL_REF_ACC`), so openpilot launches are bounded by the vehicle.

- The model calls a red light 4–5 s out (strict evidence at ~70 m / 14 m/s, need ≥ 1.7 m/s²) and its e2e request
  ramps −0.7 … −1.3 → −2.5 into the last 3 s. Force Stops committed 4.5 s (#1) and 0.5 s (#2) before the driver braked:
  #1 lost every arbitration frame to e2e because the MPC's stop column started at +1.45 and took 1.5 s to reach −1.6
  (stale free-run `a_prev` under `A_CHANGE_COST`, quadratic obstacle cost); #2 was reset by one radar frame.
- Fixes: D13 committed approach profile, D14 change-cost re-anchor on any obstacle handoff, D15 faster commit and
  tracked-lead release. Closed-loop check: the maneuver plant gained a world-fixed stop line the fake model calls 5 s
  out (`Plant(stop_line=…)`); the red-light maneuver must stop short of the line with the needed deceleration reached
  within a second and eased off at the end.
- Launch: with the SCC's ~1.9 m/s² ceiling the remaining lever is starting early (field test 1); the green-light
  cap `LAUNCH_MAX_ACCEL` on combo (1.5) is worth raising toward the ceiling.

**2026-08-29, field test 3 (live report).** Stops follow the flat, eased profile but land earlier than the owner wants.
`LATCH_SETBACK` 5 → 2 m: the committed point sits that far short of the model's endpoint; 5 m compensated for the soft
column's overshoot, which the profile no longer has (route 24 red light 1: the endpoint sat ~1.5 m beyond the owner's own
stop). `PROFILE_LANDING` 4.5 → 2.5 m: the closed-loop plant stops about a metre past the landing, so the margin sets the
stop position almost 1:1 — the driven build landed ≈ 8.5 m short of the endpoint, this one ≈ 3.5 m with the same
−0.45 m/s² last second. Calibrated on routes 25/26 against the world-fixed committed point: the car stops ~0.6 m short of it (the plant's 3.5 m
column shortfall does not exist on the real ESP/hold), so the setback is the position lever and the landing margin is
feel; the owner's preferred spot sat ~2.7 m (route 25 t=1041), ~1.5 m (route 24) short of the model's endpoint and
once ~0.5 m past it (route 25 t=1558). `LATCH_SETBACK` 3 m ⇒ the car lands ≈ 3.6 m short of the endpoint, about a
metre before the owner's usual spot.

**2026-08-29, field test 4 (route 27, combo 6743a56844).** Owner: CEM stops jittery / not confident at the end, every green
about a second slow, one stale red (t=350), a lane change into a left-turn lane stopped far back (t=374), one harsh
landing (t=1052). Traces: the harsh end and the "heavy at the end" both came from the landing — the profile's need blew up
as the landing closed and, once the profile had faded, the shaping cap's −1.2 cruise floor drove the car to walking pace;
e2e's late ramp also overtook the flat profile through `min()`. D18–D20 fix those. Green launches: our release took
0.5–0.9 s after the model's path opened, the ESP a further 1.4 s (its standstill exit, not ours); D21 trims ours. t=350:
the model itself still called the stop until the driver pressed the gas — the path opened only afterwards. t=374: D21.
t=910 landed 3.5 m short of the endpoint by the 3 m setback; left as is.

**2026-08-30, route 28 (combo 62eeab0fda, first landing law).** Owner: rough landings at t≈530/745/1740/1890, a slow green at
t≈2245. All four rough stops are lead stops and the same mechanism, seen at 100 Hz and on the car's own accelerometer (`ESP12.LONG_ACCEL`):
the plan tracked the law and the car tracked the plan (`TCS13 ACCEL_REF_ACC` = our request) down to 0.2 m/s, then the switched 0.40 floor
against the MPC's hover made the target alternate −0.40 / +0.1 per frame, the positive frames dropped the stop bit into the raw PID branch
(+0.13…+0.22 throttle at 0.1 m/s) and the clamp followed 0.1 s later — a brake / blip / clamp inside 0.3 s (8 of the route's 14 stops). The
fourth stop (t≈1890) was a stop-and-go the MPC read right from the lead's creep. Fix: the corridor form of D22. The green: our hold released
0.4 s after the model's path opened (D21 working), then the plan asked +0.05 m/s² for 0.9 s — the e2e candidate's own request, with the model's
`shouldStop` still set while its path opened, wins the `min()` against the cruise ramp; combo's launch assist needs the model's own plan above
2 m/s at 3.5 s and did not fire. Built on `combo` the same day at the owner's direction (the launch code lives there): when the path is
confirmed open, the model's own request is not negative and its plan has not committed, the e2e candidate launches on the cruise ramp under the
assist's cap (`LAUNCH_MAX_ACCEL` 1.5, tapering out by 2 m/s); the lead candidate still guards a car ahead through the `min()`. Standalone BLoTv3
has no launch assist: after a hold release the e2e candidate's stuck stop bit keeps the plan's stop bit set until the model clears it (known
gap, combo-only behavior). Also at the owner's direction the Hyundai standstill hold moved to the brake request alone, no StopReq (opendbc
`combo-blatv2-409-horizon` befe6683, promoted from the field-experiment branch): the ESP's ~1.4 s exit sequence should disappear with it;
the hold on grades and over long waits is unverified until the owner drives it. **Verdict (route 29, 2026-08-30): no-go, reverted** —
without StopReq the SCC keeps building brake pressure at standstill (brake lights on, `CF_Esc_BrkCtl` active, ESP reference 0), chasing a
deceleration it cannot measure on a stopped car; the ESP's ~1.4 s standstill-exit sequence is the price of its own hold.

**2026-08-30, route 0x2a (combo 04af1155e3: corridor + cruise-ramp greens, StopReq hold back).** Owner: landings from ~5 mph still
harsh (t≈500), greens still slow. Greens: over the route's six launches the model's path opens → our stop bit clears in 0.09 s → StopReq
drops 0.08 s later → the car moves 1.38 s after that (median), with the plan at 1.0 m/s² half a second after the clear and 1.5 at motion.
Our side is ~0.1 s; the rest is the ESP's standstill exit, the same on every launch. No upstream work targets it: opendbc master sends
StopReq exactly as we do, sunnypilot's Hyundai module too, and the one PR that listed "very delayed take off" (openpilot #33032) was
a jerk/tuning change, closed unmerged. The t≈500 landing: a stationary radar return the car drove over (tracked 3.1 → −0.1 m, lateral
−1.0 → +0.2) became the lead through radard's low-speed override at 1.1 m the moment it passed the 1 s age gate; the MPC asked −2.6 for
two frames, the law's close-lead exemption passed it, and the ESP — 0.2 s to bite, 0.7 s to release — turned it into a second of −1.9
through 1.5 → 0.5 m/s while the plan asked −1.0. Across the route's six stops the car braked harder than asked by 0.6 m/s² (median)
during the release. Fixes: radard's override gets a distance floor of 0.6 s of travel (`smooth-stops`, owner of that gate), and D23.

**2026-08-31, route 0x2b follow-up (built on the owner's go):** D24 (early kiss + smaller lift deadband + rolling-only launch frames), D25 (departure release at 0.5 m/s), D26 (CEM search release). The lagged plant lands the stopped-lead maneuver with ≤ 0.25 m/s² still on the car at 0.15 m/s (was −0.37 in the field), and the CEM replay gates above. *(2026-09-25: that was the old plant's `min(plan, −0.12)` stand-in; on D36's plant — stock LongControl, the CAN-identified actuator — the first build carried −0.341 there and the maneuver was red; with D37's measured speed and withheld stop bit it carries −0.178 and is green.)*

**2026-08-31, route 0x2c (combo a56860fc62).** Owner: one stop crept at the end (t=727: D27), a curve exit after a gas override was pulled back down mid-corner (t=885: the curve branch's post-override grace), and two rolling red→greens kept braking after the road opened (t=1105/1135: D28; the standstill-gated launch boost never arms on a rolling green — the cruise ramp is the recovery, ~0.8 s).

**Open items (known, not fixed — recorded 2026-09-17).** Seen in the logs, no fix built; the branch should not read cleaner
than it is.

- **2026-09-15, route 0x7e — the big model plans through an anticipated green.** Force Stops' open-path release and CEM's
  open-road evidence (`stop_release_open`) judge the plan by its end (terminal speed, raw path length), so a green the model
  expects far down the road reads as an open road and the car gave up a red-light stop at 16 mph. No fix built: the
  2026-09-17 stop-evidence guard on the open-path release narrows the hold case only. **2026-09-25:** the release there was a
  moving commitment at 9.6 m/s, 26–30 m from the line, with `shouldStop` off, so that guard does not touch it; the README's
  2026-09-17 entry credited the guard with this route and is corrected.
- **2026-09-02, route 0x3e t=577 — a queue inch trips the lead-departure pre-release.** A lead moving 0.61 m/s for ~1 s
  cleared `LEAD_DEPARTURE_SPEED` (0.5), the pre-release launched the car and it had to stop again. Proposal: confirm a
  departure by growing gap or sustained speed, not one instantaneous reading. Not built.
- **2026-09-02, route 0x3a t=557 — an uphill crawl never arms the anti-creep press.** Crawling at 0.2 m/s for 4 s behind a
  stopped lead, the D27 press never armed: the landing latch needs `a_target` strictly below −`KISS_DECEL` while the raw plan
  sat at exactly −0.15. Proposal: press whenever `v ≤ KISS_SPEED` with stop intent. Not built. **2026-09-25:** the press is
  gone (D34). Replayed open loop, the landing was latched through this window after all, and the press was off because the
  raw MPC plan was positive (+0.8) while the car re-accelerated 0.12 → 0.23 m/s; the anti-stall integral presses from
  t=557.3. The latch-entry explanation above may be only part of it; the closed-loop run of this window is still to do.
- D30 and D31 are **field test pending**, as their §2 rows say, and so is every fix in the 2026-09-17 audit pass.

**2026-09-25 — the braking audit and its step 1 (no drive).** The 2026-09-24 audit of the branch's braking — stability,
accuracy of the calculations, runtime — kept 71 findings and refuted 11. Nothing in it was driven: every effect is an
open-loop replay of the logged drives, a closed-loop replay through a fitted ESP, a plant run or a unit test. Its verdict:
the instability is in the plan, not in the car. A lead-free red light is braked by five owners in turn — a set-speed −1.2
that a per-frame stop tier switches on and off, e2e's action, the MPC stop column's overshooting first solve, the profile
planning to a point the model still reads long, then several hand-overs in the last 3.5 m/s — and every hand-over is a step
or a reversal of the request: 14 of 15 model-led stops show hard, soft, hard on the logged plan, while lead stops through the
same ESP have a median largest step of 0.10 m/s² against 0.73. Not the cause: LongControl (no loop on this car), the ESP's
slow release on its own, a body mode, the 1.0 s follow time, runtime. Recommended path: one law owns the stop from the first
evidence to the kiss (design B), in five gated steps, the first a set of independent root fixes. Owner, 2026-09-25: "I agree
with all of your findings and recommendations. Build it." Step 1 first, gated, then merged to `combo` for his drive; steps
2–5 wait for his seat. Step 1 was built in two rounds the same day: the first build's recheck left six items open (a
lead-leave collapse, a red maneuver, two commit steps, F011's held-out failure, a softer lead hand-over, the landing's own
exit), and the second round closed or recorded each — D32's told-target hand-over and D33's clip, D37 and D38 built, F011
kept out, the hand-over residuals recorded in D32.

- **Shipped in step 1: D32–D38.** Gate on the integrated tree, both `ACCEL_MAX` pins: the controls unit tests 412 passed, 1
  skipped, 1 strict expected failure; the maneuver suite 59 passed, 6 strict expected failures (each with its cause at the marker),
  0 failed; selfdrived passes but for the two offroad-alert tests that fail on the tip too; 23 one-change reverts, each caught by
  1–110 tests. The six traced stops (0x62 t=3159, 0x58 t=382 / 558 / 143, 0x5e t=155 / 752), open loop: reversals of 0.3 or more
  28 → 20, largest step 1.022 → 0.360, steps above 0.2 26 → 2 (both CEM entries), the commit-frame step −0.100 on 9 of 10 field
  commits (0x59 t=608.8 −0.396, its converged need), leave-class up-steps 0, no new moving release and one fewer (D35). Closed loop
  over three ESP fits: plan reversals 30 → 20 (car reversals 12 → 12), median largest step 0.700 → 0.182 (max 1.126 → 0.449),
  median peak −2.40 → −2.36, the car at 0.4 m/s a median −0.72 → −0.37, rest a median 0.15 m further short (per stop
  −0.12…+0.24 m); the largest steps left are the e2e re-join at the landing and the CEM entry (steps 3 and 4). 21 routes, whole
  route, open loop: moving up-steps above 0.3 82 → 24, alternations above 0.15 189 → 6 (six gate routes) and 153 → 13 (15 held
  out), leave-class up-steps 13 → 0, landing exits above `J_CRUISE`·dt 87 → 0, lead-appearance edges better or equal on all six
  gate routes, stop-bit releases within 0.5 s of before on 59 of 60 and 43 of 44 (the rises move to the standstill speed by design,
  D37), the emergency lead-brake windows within one frame.
- **Not in step 1.** F011's plan-physics green (D35: built, held on its held-out gate; whether an anticipated launch is a green is
  the owner's call at step 4). F045, the set-speed reduction on the comfort law in both modes: landing it while the pre-commit cap
  still rides the set speed would silently neuter the cap and move braking late, so it lands with step 4.
- **Open at merge (the 2026-09-25 round-2 recheck).** (1) The profile joining from the published target can evict an e2e request
  that set it, which re-joins with a 0.56–0.70 brake step on commits the model ramps into at 2.5–4 m/s³ (D37); the incumbency
  clause that removes it is built and measured, not in this tree. (2) The stop column's need at the true speed steps the commit
  where the profile does not take it: the 22 m/s maneuver (0.208, a strict expected failure) and 0x59 t=608.8 (−0.396) — step 3.
  (3) F011 held: 0x62 t=3151.56's false moving green releases in every replay, and with D37's entry its let-off is as deep as the
  braking had reached (D35). (4) D32's hand-over residuals (R7.5), among them a 2–6-frame radar dropout while braking hard, which
  lets off 0.27–0.74 further than shipped on frames 2–6. (5) D37's trades: lead following pivots more, lead stops rest 0.3–1.3 m
  farther back (D3 goes back to the owner), and at standstill behind a standing lead the MPC's true-speed hover meets the stop bit's
  threshold (F009's cause). (6) A car braking 0.3 m/s² short rests about 3 m behind a stopped lead (a strict expected failure; the
  anti-stall integral's design point, step 3). (7) The landing's residuals (D34): the press's one-frame drop at 0.10 m/s, the lead
  requirement's one-frame let-go as a lead starts to move, the raw-radar dropout lift, and creep 0.5 re-accelerating 0.054 against
  0.05 in two gate runs. (8) After a Force Stops green, a candidate still braking below the kiss on the next frame re-latches the
  landing at rest and nothing ends it — F035's launch gap, pinned by a strict expected failure, 0 of 77–82 field standstill exits.
  (9) The seed-switch dips at `ACCEL_MAX` 2.0 (D33). CI's unit runner runs none of the fork's 312 pytest-style unit tests (§6), so
  a green CI on `combo` covers the maneuver suite only: run the suites by hand before any push.
- **Merging to `combo`.** `combo`'s planner takes the same four edits: the speed filter removed, the landing carry removed, the
  stop-bit block (with `combo`'s whole OR, the launch assist's e2e bit included, in its `else` branch) and the `StopLanding`
  constructor. There the withheld stop bit keeps LongControl in pid through the landing, so `smooth-stops`' settle no longer runs.
- **D30's first field data, recorded unjudged.** 0x5e and 0x62 ran D30 builds. On the three traced D30-build stops the
  follow-down D30 added above `DOWN_SPEED` moved the point in over the last 3–4 s and the need rose after the commit: 0x5e
  t=155 1.82 → 2.77 m/s² (the logged peak −2.75 about 7 m out), 0x5e t=752 1.57 → 2.38, 0x62 1.53 → 2.13 with the point moving
  3.7 m in 1.85 s. A frozen-point counterfactual on the same speed traces keeps the need flat or falling, but would have left
  the point 3.6 and 5.4 m beyond where the car rested: D30 corrects a placement error and pays for it late. The structural fix
  is step 2's point estimator (F004). On 0x62 t=3151.56 D28's moving green released the commitment falsely (the plan's terminal
  speed hovering across strict's 1.0 m/s 49 m from the line, no `shouldStop`); it re-committed about 1 s later. D30's verdict
  is the owner's; it stays field test pending.
- **What waits for the drive.** Step 1 on `combo`: lead-free CEM red lights from 12–22 m/s, a yellow, a green mid-approach, a
  green from a hold, walking-pace queue stops behind a stopped lead — one where the lead moves off before the car stops among them —
  a close cut-in at walking pace, a lead leaving at speed, a CEM exit while braking, a grade stop and a long red-light wait. In the
  logs: the published `aTarget` at the commit (no step above 0.2 — 0.1 where the profile takes the frame — unless the need is real,
  no ring), no let-off and re-brake of 0.15 or more on the approach, cruise hand-overs and landing exits at most `J_CRUISE`·dt per
  frame, launches leaving −0.15 at 0.10 per frame, `aTarget` still rising through a lead departure, `aTarget` and `aEgo` at 0.4–1 m/s
  and at the wheel stop, the stop bit and StopReq rising at 0.10 m/s or below and held at rest with no creep behind a queue, the final
  roll time and the rest gap at lead stops, the rest point. Then, each gated on his seat: step 2, a stop-point estimator replacing
  the frozen committed length and D30's follow counters, with a commitment surviving invalid model frames (F004, F010); step 3,
  `stop_law.py` owning a moving commitment with no MPC stop column while it drives, the corridor inside the law, e2e pulling
  continuously and `E2E_STOP_MARGIN` deleted (F015, F024, F039, F040) — and with them the column's need step at the commit and
  the e2e re-join; step 4, the law from the first evidence, the episode persisting while the plan still slows, CEM holding its
  mode on it and the pre-commit cap deleted (F002, F029, F014, F047, F012), with F045 and the F011 question; step 5, a lead owning
  the stop only if it rests before the stop point (F032). His questions once step 1 has been driven: design B or today's structure
  with every root fix, the target shape and the entry level; D3's 7 m lead-stop geometry, now with lead stops resting 0.3–1.3 m
  farther back than before, and `KISS_SPEED` as car-side values (D34); D37's lead-following trade; retiring `combo`'s settle (D37);
  whether an anticipated launch is a green (D35).
