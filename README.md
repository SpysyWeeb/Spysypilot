# BLoTv3

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress — phases 1–3 implemented and replay-gated against BLoTv2; merged into `combo` 2026-08-29 (replacing BLoTv2) for the owner's field test.**

## What it does

BLoTv3 is the rewrite of the Palisade longitudinal tune
([BLoTv2](https://github.com/SpysyWeeb/Spysypilot/tree/BLoTv2)). The road behavior BLoTv2 got
right is kept — the continuous cubic acceleration envelope, ordinary-cruise comfort shaping, the
necessity supervisor that softens the MPC's jerk cost and pads following time only when a lead
requires it, radar-anchored model lead trajectories, lead-departure pre-release, Conditional
Experimental Mode for lead-free model stops, and Force Stops' committed stop point — but every
longitudinal decision gets exactly one owner in exactly one process, the code takes the shape
upstream uses for the planner, and the defects found in the 2026-08-28 review of BLoTv2 are fixed:

- Conditional Experimental Mode could only ever auto-engage when the radar saw **no** vehicle at
  all (entry evidence was wiped on every control tick a lead existed, at any distance).
- A committed stop had no owner once the car stood still, so `shouldStop` followed the model's
  flicker and could pulse the car's own ACC resume.
- The supervisor's stand-down raised the FCW "BRAKE!" alert in a single frame with no vision
  confirmation; the FCW distance check silently moved from radar extrapolation to the model path.
- Any pedal tap switched a manually enabled Experimental mode off for two seconds.
- The headway pad snapped to zero exactly as the required deceleration crossed 1.5 m/s², and the
  low-speed jerk hold only engaged if the scale had reached its exact floor.
- The turn budget was raised to a flat 4.0 m/s² and never limited anything; the `longcontrol.py`
  clamp could not bind in either opendbc configuration; the third-lead "ponytail" reset the
  supervisor's policy while lead0 was still the nearest car.
- `force_stops.py` existed in three divergent copies (`force-stops`, `BLoTv2`, `combo`).

Since the 2026-08-29/30 field tests: the committed approach profile (tapered landing), the change-cost
re-anchor on any obstacle handoff, faster commits, green release in three frames, lane-change re-qualification, and the
**landing law** — the planner's last word on every stop's final metres: a corridor (allowed braking 0.70·v + 0.30 m/s²,
a creep floor, both tapering to a 0.15 m/s² kiss at walking pace) that latches through the MPC's hover and through standstill
until a real launch; lead physics never blocked. The owner's original June Smooth Stops design rehomed in the planner. Since
2026-08-31 the kiss arrives a full ESP-release-lag before the wheels stop (0.40 m/s), the lead-departure pre-release needs 0.5 m/s,
and Conditional Experimental Mode releases its stop search when the model's own path shows the road open — a lone borderline hint
can start the search but not keep it alive past a green light. Route 0x2c polish: a moving commitment now releases within three
frames of the road opening, like the hold — it used to brake 1–2 s past a rolling green. Since 2026-09-25 the corridor describes
what the car should carry and is requested one actuator delay (0.55 s) ahead of it, and neither edge reads the measured
deceleration: the release lift and the anti-creep press of 08-30/31 are gone, and one anti-stall integral — the one place the
measured deceleration enters — presses a car that is not slowing (never at standstill). Whatever the landing held when it ends is
handed back at the cruise jerk, and while its kiss lands a rolling car the plan's stop bit waits for the standstill speed.

## How it works (planned)

- **selfdrived** keeps owning the effective Chill/Experimental mode, exactly where stock publishes
  it. `ConditionalExperimentalMode` shrinks to that one job: mode request in, no stop fields out.
  It still steps every control tick (so a hung model still releases the mode) and only looks at
  new model frames for evidence.
- **plannerd** owns everything else. `stop_helpers.py` is one stateless classifier of model stop
  intent, imported by both processes, so the only signal that crosses processes is stock's
  `selfdriveState.experimentalMode`. `force_stops.py` becomes the sole owner of shaping →
  commit → **hold through standstill** → release; the four `conditionalStop*` cereal fields are
  retired. `necessity_supervisor.py` (BLoTv2's `blotv2.py`) keeps its triggers and fixes the two
  cliffs. `longitudinal_lead.py` is the one definition of a usable lead and anchors the model
  lead once per frame. `long_mpc.py` gets one `update()` call that resolves obstacles, sets the
  cost weights once, solves, and scores FCW against the trajectory it actually solved.
- `radard.py` and `longcontrol.py` are not edited; `controlsd.py` carries exactly one line, D31's
  reordering (`actuators.longControlState` is read after `LoC.update()` computes it). Stock opendbc limits apply on
  this branch (2.0 m/s²); `combo`'s opendbc/panda lineage supplies the 4.0 m/s² envelope.
- Design, owner decisions, module contracts, the hold state machine rules and the acceptance
  gates live in [`docs/BLoTv3.md`](docs/BLoTv3.md). Phases, each field-tested by the owner before
  the next: (0) branch, docs, harness, replay tooling; (1) cruise layer; (2) lead layer;
  (3) stop layer; (4) staged integration into `combo`.

## What changed

**2026-09-25 — audit step 1.** A braking audit (2026-09-24) found the stops' instability in the plan, not in the car: a lead-free red light is braked by several owners in turn, and every hand-over between them is a step or a reversal of the request. The owner approved all of its findings. Step 1 is the set of independent root fixes, each with its own tests; steps 2–5, one law owning the stop, wait for his drive. Nothing here is field-tested: every number below is a replay, a plant run or a unit test.

- **The jolt when a stop commits.** When the MPC's stop column took over from the free run, its first solve still started from the free run's plan, climbing toward the acceleration ceiling, and overshot — −1.92 where the converged answer was −1.26 — then rang for half a second; D14 had re-anchored only the change cost. A hand-over now restarts the solver from the target the car was last told and takes one extra iteration, and it is keyed on what the obstacle is — the free run, a radar lead, the stop column — rather than which column holds it: a lead changing slots or sitting on the stop line is not a hand-over, and a radar dropout of a few frames is not a departure. Every field commit's first solve is now within 0.001 of converged.
- **Surges when a braking owner left.** The cruise candidate kept slewing on its own value while something else set the output; when that owner left — Experimental dropping, a stop released, the approach profile ending — the output jumped to the stale value, +2.29 m/s² in one frame at a mode exit. Cruise now takes over from the published target at its own jerk. The MPC is also seeded with its own plan while it drives, clipped to what the car can do, instead of the published target, which is the plan read 0.55 s ahead and came back as a 10 Hz ring.
- **The landing law's loops.** The release lift (a proportional loop on the measured deceleration against the raw plan, which rang with the MPC), the 5 m close-lead switch, the anti-creep press and the stall watchdog are gone. The corridor now describes what the car should carry and is requested one actuator delay early; the braking a lead needs always passes — stopping 7 m behind it, or, for a closer one, short of it; one anti-stall integral, not applied at standstill, replaces the press and the watchdog. Landing request reversals of 0.3 m/s² or more fell 18 → 8 on six routes of open-loop replay.
- **The landing's exit.** The landing let go of whatever it held in one frame when it ended — the kiss at a launch, the floor and the anti-stall press when the plan climbed out of it. It now hands that back at the cruise jerk: over 21 routes of replay the largest exit step fell 0.84 → 0.10 m/s², and launches from a hold or behind a departing lead start as before.
- **A committed stop dropped at walking pace.** The model's stopped plan ends a few centimetres behind the car, and that non-positive endpoint reset a live commitment — 13 times on 30 routes. A commitment now keeps its point through it and forms the hold.
- **The car's own speed and the last metre.** The MPC planned from a filtered integral of its own commands instead of the car's speed; with nothing below the planner closing a loop on speed, whatever the car really did became phantom speed — up to 0.45 m/s in a stop's last metre — and in the plant a car braking 0.3 m/s² short of every request drove into a stopped lead. The MPC now plans from the measured speed: that car stops about 3 m back, and behind a stopped lead the car eases the last 0.3 m/s more gently; where it comes to rest moves by up to about a metre, in either direction depending on how the car's brakes really answer (the landing test bed: 0.3–1.3 m farther back; the closed-loop replay through three fitted brake models: 0.9 m closer to 0.3 m farther back), to be re-measured on the drive. And the stop bit no longer hands a rolling car to LongControl's stopping ramp at 0.3 m/s while the landing's kiss is still landing it: it waits for the standstill speed (0.10 m/s), or for a lead nearer than the kiss can stop short of.
- **The approach profile's first frame.** The committed profile entered from the car's measured deceleration, which lags the request and rings; on two field commits that was a one-frame brake step of 0.55 and 0.20 m/s². It now enters from the published target at its own 2 m/s³: 9 of 10 field commits step 0.10, and the tenth is the stop column's real need (0.40). This reverses D13's entry.
- **Standing behind a lead (round 1b, corrected in steps 1c and 1d).** Planning from the true speed made the MPC's own stop bit drift across its threshold while the car stood behind a car that was not moving, so the bit dropped and re-rose and the car could inch forward (0.17–0.36 m in simulation). Once the MPC has stopped the car behind a lead, the lead-departure release is now the only thing that lets it go; a release taken back while the lead still stands where it was released (within the radar's two 0.1 m range steps a standing car reads) holds the car again at once, landing included, while one taken back after the lead has moved leaves the launch to the MPC's plan; and a measured lead speed counts only on the radar track the model confirms, so radard's close-range return of a standing car no longer releases it. All of it belongs to the car it came to rest behind, and only evidence that it has left ends the hold: its departure, a car standing beyond where it stood taking its place (one frame of it is enough; the car then closes up to that car), or, while it is missing, the model seeing open road the way Force Stops sees a green at a red light — three frames of a long open plan, or its launch reading held for about 0.4 s; a one- or two-frame flash is not one; otherwise the driver. A car crossing in front at a red that is flagged as the car comes to rest is held for like a lead: if it clears on a flash of open road the car stays, and once it has cleared and the model has seen open road for three frames the car goes — at the green, on the frame it would have gone without that car. A crossing car flagged later is not the car's lead, but while it is flagged it still ends Force Stops' hold, and if it clears on the very frame of a one- or two-frame flash the stop bit drops for those frames (a Force Stops item, open for the owner). A lead lost to radar and camera while the model still plans the stop, for any time, or a car cutting in between, keeps the car held, and a car held at rest plans its launch from rest, so a lead that went missing and came back leaves nothing in the launch. Over 31 routes of replay the time the car stood released with no departure confirmed fell from 68.5 s (15.7 s before step 1) to 6.7 s; of its 97 frames with a request above +0.1, 79 are behind a lead that had already moved (a departure, or a queue creeping on). The cost: when the model forecasts a departure a second or more early and the forecast wavers while the lead still stands, the hold comes back for 0.2–0.95 s before the release — an extra stop-bit cycle in 6 of 151 standstills behind a lead, and a later release on 11 of 115 departures (median 0.5 s); that and three real releases 1–4 frames later are open for the owner (D39). A car whose lead vanishes for good waits until the model sees open road or the driver acts; after a slow roll-up with no landing it then launches on the plan's own no-lead acceleration (+1.75 m/s² on its first frame in simulation, against +0.03 behind a lead that drives off), and whether to smooth that is open for the owner (D39).
- **The Chill throttle gate (round 1b).** Chill caps cruise at the coast while the model's gas-press probability is at or below 0.4, and it decided on every frame's reading, which near 0.4 flips back and forth: 16.7 times a minute, a small saw-tooth in the cruise target before slowdowns. It now decides on the probability's 0.25 s mean: 3.0 flips a minute and cruise reversals 738 → 98 over 22 routes; a real denial takes effect 0.15 s later (median), and the car coasts a little less before a red (a median 0.02 m/s). The gate is upstream's: carried here until upstream takes it.
- **The maneuver plant is the car.** BLoTv3's stop maneuvers run on Palisade CarParams with stock LongControl, the car's speed filter, the radar's 0.1 m distance steps and a CAN-identified actuator, at both `ACCEL_MAX` pins; red lights at 14–20 m/s assert on the plan (no step above 0.15 m/s², no let-off and re-brake), and new maneuvers cover a car that brakes short of its requests and a lead leaving a latched landing. The previous code fails them; this step passes, with six strict expected failures, each with its cause.

Held back: a plan-physics green release for moving commitments — built, but on held-out routes it released stops the model expected to relaunch from; whether such a stop counts as a green is the owner's call at step 4. The set-speed comfort law in both modes lands with step 4. Known open at merge, recorded in `docs/BLoTv3.md` (D32–D38, §8 2026-09-25): the 22 m/s red light and one field commit step at the stop column's real need (step 3); a radar dropout of a few frames while braking hard lets off more than before; lead following corrects the car's real deviation and pivots a little more; a car braking 0.3 m/s² short rests about 3 m behind a stopped lead. CI's runner collects none of the pytest-style unit tests: run the suites by hand before any push. A model braking hard into a commit keeps its place against the joining profile (the incumbency clause is in step 1's commit; the step-1 notes listed it as not in this step). Also corrected: the 2026-09-17 entry's route 0x7e attribution and the D30 entry's follow-down wording (below), and this README's description of the landing law and of the approach profile. Field test pending.

**2026-09-17 — audit fixes.** A cleanliness audit of the branch found five logic defects in the stop layer, none of them covered by a test. All are fixed with a test each; nothing here is field-tested.

- **Force Stops' slow release left the last commitment behind.** When the detector decayed below `RELEASE_THRESHOLD` with the position hold spent, the commitment ended without clearing the approach profile's anchor or the green-release frame count, so the next, unrelated commitment started its jerk-limited ramp from the old braking level instead of the car's own acceleration — a brake step on the first frame, which is not what D13 says the profile does. Every release now ends the commitment whole.
- **The hold's green release ignored the model's own stop call.** D28 gave a moving commitment's fast release a no-stop-evidence guard and called it "the same release the hold got in D21"; the hold's counter never had it. A frame with a path longer than `RELEASE_OPEN_LENGTH` and `shouldStop` still set dropped a hold in three frames. The two releases are now literally the same test. *(Corrected 2026-09-25: this entry first tied the defect to route 0x7e and said the hold could not re-form. On 0x7e the commitment was still moving, at 9.6 m/s with `shouldStop` off, so this guard does not touch it — it stays an open item in `docs/BLoTv3.md` §8; and a hold let go this way re-forms once the model's path is short again.)*
- **A gas tap at speed armed nothing.** Only a tap that broke a *hold* armed the `REARM_S` re-entry window; a tap that broke a moving commitment did not, so reaching the line inside those 10 s formed no hold and the car sat on the raw stop bit until the gas grace ran out. A tap now arms it either way, as a lead does.
- **A radar dropout hid an invalid model from CEM.** Completeness was judged on model *and* radar validity, so a radar dropout certified a garbage trajectory as complete and the `MODEL_INVALID_RELEASE_S` release never opened (the mode still left through the clear hysteresis, ~5 s instead of ~0.5 s). The two are separate inputs now: completeness is the model frame's alone, an invalid radar only empties that frame's stop evidence and revokes a pending lead release.
- **A creep resume read as a launch.** A creep or a grade rolling the car past `RESUME_SPEED` turns a hold back into a moving commitment with the latch alive; the planner derived its landing-corridor `launch` from the hold bit alone and tore the corridor down mid-stop for a frame or two. A launch is now a corroborated lead departure or a real Force Stops release — the frame where the commitment itself ends, which is the frame that carries no stop point.

Hygiene in the same pass: a non-finite plan target passes the landing corridor untouched instead of poisoning its clamps; the near-stopped-lead pad's divisor is now the named `STOPPED_LEAD_FULL_DECEL` (1.2 m/s², BLoTv2's field value — the docs claimed 1.5, which is only the stand-down gate); a hold that rolls again starts its release and follow evidence over; new tests for D31's publish order, D3's 7 m `STOP_DISTANCE` and the maneuver harness's launch-band-scoped `ensure_start` (ported from `combo`); this README and `docs/BLoTv3.md` corrected against the code; the plan-source legend in both tuning layouts gained `stop`. Field test pending.

**2026-09-06 — D31, the published control state.** `controlsd` read `self.LoC.long_control_state` into the CarControl before `LoC.update()` computed it, so the state was always one control frame behind the acceleration published beside it. The Hyundai interface derives the stop request and the standstill-exit jerk limit from that field. Now published after the update. Expect a process-replay diff wherever a stop or a launch crosses a frame boundary. Field test pending.

**2026-09-06 — D30, the committed point follows the model on confirmed evidence.** Two stops (routes 0x58 t=548, 0x59 t=609) rested 3–4 m past the model's settled endpoint while the day's good stops rest ~1.7 m before it. In one the commit took the first strict frames' endpoint, 5.8 m long, and the follow-down waited for 3 m/s; in the other the forward follow chased a 1.5 s endpoint excursion of +8..+14 m during the braking onset and moved the point 4 m with no way back above 3 m/s. `FOLLOW_CONFIRM_S` (1 s of evidence past the deadband, drained only by contrary frames) now gates the forward follow at any speed and the follow-down above `DOWN_SPEED`; below it the follow-down stays immediate. Open-loop replay puts both committed points back at the good stops' placement (−3.0 / −2.8 m vs +0.8 / +2.5 before); route 25's stuttering drift extends 0.9 m less. *(Corrected 2026-09-25: before D30 there was no follow-down above `DOWN_SPEED` at all — the latch was immune to a collapsing endpoint there; D30 added one, gated by `FOLLOW_CONFIRM_S`. On the three traced stops of its first drives it moved the point in over the last 3–4 s and the need rose late, by up to about 0.95 m/s²; `docs/BLoTv3.md` §8 records that data, unjudged.)* Field test pending.

**2026-09-02 — D29, the pursuit tail.** Braking for a truck that then drove off (route 0x3b t=390–405), the supervisor's low jerk cost switched off at the plan's zero crossing, exactly where the MPC had to swing to acceleration; it now stays for 3 s after excess braking behind a lead that is accelerating away. Replay on that event and the supervisor tests gate it; awaiting the owner's drive.

- Phase 0 (2026-08-29): branch cut from `stock` `511f2b60b4`; `docs/BLoTv3.md` added; the
  longitudinal maneuver harness got a real `all_checks()` shim, radar/model validity schedules,
  `selfdriveState.enabled`, and three distinct, fully populated `leadsV3` messages.
- Phase 1 (2026-08-29) — `openpilot/selfdrive/controls/lib/longitudinal_planner.py`: the cruise
  acceleration ceiling is the continuous cubic envelope `0.6 + 3.4 (1 − v/40)³`, clamped by the
  deployed opendbc `ACCEL_MAX`; the jerk schedule is `[2.0, 1.6, 1.0, 0.6] m/s³`; ordinary Chill
  cruise above 15 m/s uses the proportional comfort target (5 mph ≈ 0.40 m/s², pitch-compensated
  coast on reductions, blended in from 8 m/s) whenever the radar is healthy; there is no lateral turn budget (removed after the
  2026-08-29 field test — accelerating out of curves felt held back); the MPC's acceleration-change
  cost stays on through standstill. Behavioral tests in
  `openpilot/selfdrive/controls/tests/test_longitudinal_planner.py`. The e2e candidate only enters
  arbitration with a valid model, as in BLoTv2. Replay against BLoTv2 on routes d7, d9 and d2:
  every cruise-candidate difference was the (since removed) turn budget, and the three 75↔80 mph corrections on
  d9 settle at the same +0.40 / −0.40 / +0.40 m/s² in both; lead-candidate differences are
  phase 2's. Measured cost of D4 behind a departing lead: the first step is 3× gentler and the
  launch reaches half its peak at 1.35 s instead of 0.70 s — a phase-2 field item.
- Phase 2 (2026-08-29, branch `BLoTv3-phase2`) — the lead layer. `longitudinal_lead.py`: the one
  definition of a usable lead (`LeadObservation`, `lead_present`, `relevant_lead`) and
  `anchor_model_lead`, which validates the model's lead forecast once per frame and anchors it to
  radar (tolerating the few cm/s of below-zero noise a stopped lead reads on both sensors — the
  strict gate dropped the anchor mid-launch in the 2026-08-29 field test; reversing still fails closed). `necessity_supervisor.py`: BLoTv2's supervisor with two fixes — the following-time pads
  hold at their ceiling instead of vanishing once the required deceleration passes the stand-down gate
  (the onset pad ramps to 1.5 m/s² of lead braking, the near-stopped-lead pad to 1.2 m/s² of required
  deceleration), and the low-speed hold keeps whatever softening was built while necessity-braking (a
  stand-down or lead loss still releases it); its stand-down never reaches an alert.
  `force_stops.py` also owns the committed approach profile (field test 2, 2026-08-29): the constant
  deceleration that lands short of the committed point, entered from the car's own deceleration at 2 m/s³ (from the
  published target since 2026-09-25, D37), capped at 3 m/s², fading out from 3 m/s to gone at 1.5 m/s so the MPC column and the
  hold land the car — in that field test the MPC's stop column alone swung from +1.45 to −1.86 over 1.5 s after a commit
  that needed 1.9 m/s².
  A commitment forms after 0.3 s of strict world-fixed
  evidence and only a tracked lead breaks it; the MPC re-anchors its change cost on every obstacle handoff.
  `long_mpc.py`: one `update()` call takes
  the anchors, the supervisor's jerk scale and pad, and a committed stop point; it sets the cost
  weights exactly once, applies the policy only while lead0 owns the solve, re-anchors the
  change cost on a handoff, and has no third-lead machinery; `STOP_DISTANCE` is 7 m and the
  aggressive personality follows at 1.0 s as in BLoTv2. The planner wires the two and the
  lead-departure pre-release. `LongitudinalPlanSource.stop` is added to cereal. Replay against
  BLoTv2 on route d7 segment 0: every lead-candidate frame now matches to the float; only the 64
  turn-budget frames from phase 1 differ.
- Phase 3 (2026-08-29, branch `BLoTv3-phase2`) — the stop layer. `stop_helpers.py`: one stateless
  classifier of model stop intent (every stop tier, guard and constant in one place, typed capnp
  access), the launch-evidence test, and the corridor rule that fails closed unless every lead
  hypothesis with any probability is outside the stop path. `force_stops.py`: rewritten as the sole
  owner of shaping → commitment → **hold through standstill** → release, fed only by that
  classifier and `carState`; the mode gates entry only, a standstill flicker on a grade drops back to
  a commitment rather than to nothing, a relevant lead or a gas tap releases the hold and it re-enters
  as soon as the car is stopped with stop evidence again, launch evidence or a mostly-clear, moving,
  lead-free 4 s window releases it, and its `holding` output forces `shouldStop` in the planner.
  `conditional_experimental_mode.py`: mode request only, still stepped every control tick (a hung
  model releases within 0.5 s), evidence judged on model frames; a raw lead on a control tick now
  revokes only a pending recent-lead release instead of wiping entry evidence, so a distant vehicle
  no longer blocks the handoff. `selfdrived.py`: a small hook resolves the manual setting and the
  conditional request; a pedal tap no longer switches a manually enabled Experimental mode off. No
  cereal fields are added; the planner reads the stop point from Force Stops directly. Replay against
  BLoTv2 on d7 segment 37 (the route's Experimental-mode stop): planner and mode transitions identical.
