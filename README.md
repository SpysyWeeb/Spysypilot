# Spysypilot daily

A minimal-diff personal branch on the official master baseline (code as of the Mountain Dew release), with the Chestnut big model pinned to Cinque v3 (see the Chestnut section). Experimental controller changes remain subject to owner/device/field validation; local tests are not a driving-safety qualification.

## Selected features

- **Torque + SOL:** default classic-Hyundai 409 / +4 / -7 envelope, retaining existing lower-limit/CAN-FD exceptions and paired safety. SOL provides independent lateral operation, fault/freshness and monitoring integration. Driver steering caps an available SOL torque request at +/-0.5; smaller ordinary requests stay smaller, and downstream limits still apply.
- **Nudgeless lane changes:** a turn signal can initiate the existing lane-change maneuver without a steering nudge, retaining speed and blind-spot gates. The first start, automatic or manual, consumes that signal event's automatic allowance. Holding the signal does not grant another automatic attempt; another manual nudge is still permitted, and cycling the signal rearms automatic operation. Braking cancels the automatic allowance for that signal event. This does not add release-and-nudge edge detection or prove a lane line was crossed.
- **Lane centering:** the feature branch's correction, filter, confidence/geometry checks, one-line width hold, and live toggle are imported without changing their behavior. The deliberate-plan fade reads where the plan is heading, so a plan that has started to move over, such as into a lane shift around cones, is no longer fought at full strength. A driver push with torque the way the model leans, moving the plan at least 0.1 m further and leaving it at least 0.25 m off center, stands the correction down until the plan returns within 0.15 m of center, 120 s pass, a blinker or lane change, lateral inactivity, or a push back toward center (in progress; not field validated). The correction feeds the existing curvature limits and controller, then SOL's downstream driver-input cap. It also operates during SOL-only lateral activation.

- **ESP-active debounce:** the common car-event path raises `espActive` after 50 consecutive active samples (nominally 0.5 s). Any inactive sample resets it. This delays genuine interventions as well as nuisance pulses; it is not a bump detector. `espDisabled`, other faults, the existing downstream soft-disable policy, and vehicle/panda safety behavior are unchanged.
- **Hot-swap Experimental:** hold the distance button for 0.5 s to toggle the saved manual Experimental preference once per hold; a short tap still cycles personality. Requires openpilot longitudinal control. The hold action is disabled on Hyundai CAN-FD, whose edge-only button stream can pause long enough to misclassify a tap; release still cycles personality there. The classic-CAN Palisade is eligible. Effective mode remains manual OR Conditional Experimental Mode, so manual-off does not veto an automatic CEM request.
- **Swapped cruise-speed steps:** for openpilot-owned set speed, short taps move to the next five-unit grid point and held presses repeat one-unit steps. Metric/imperial conversion, speed bounds, engagement, gas-override and standstill-resume guards remain. Vehicle-owned cruise speed is unchanged.
- **Initial Max Speed setting** (Settings → Toggles; shown only with openpilot longitudinal on a non-PCM-cruise car): the set speed on engage is the larger of current speed and this setting, clipped to 40–145 km/h; resume still restores the previous max speed. Default 40 km/h = unchanged engage behavior until raised. Upstream #39081 is taken alone; its UI follow-ups (#39099, #39098) are not, because they depend on A New Dawn (upstream's Experimental Mode removal), which daily does not take since it keeps Experimental Mode for hot-swap and Conditional Experimental Mode.
- **Smooth Stops radar filter:** the current branch adds an age/range gate only to unconfirmed low-speed lead overrides: 20 consecutive radar updates and a distance above `max(0.75 m, 0.6 s × ego speed)`, retaining the existing below-4 m/s, lateral and 25 m gates. A closer qualifying track replaces the current lead whether or not that lead is vision-confirmed; a missing radar cycle restarts track age. This can also delay or reject a real newly appearing, flickering or close unconfirmed object. BLoTv3's landing and the existing LongControl remain unchanged.

- **BLoTv3 longitudinal (in progress; not field validated):** full model-lead forecasting, following-response supervision, cruise comfort/tuning, Conditional Experimental Mode, committed model stops, lead hold/departure handling, planner handoffs and final stop landing. Newer upstream AccelBoost is retained. The host/Hyundai acceleration ceiling is 4.0 m/s² with matching native Hyundai safety; speed-dependent planning, jerk and per-platform limits still apply. The nominal stopped-lead distance is 7 m and aggressive base following time is 1.0 s. This replaces stock longitudinal planning, not radar selection or steering. See [BLoTv3 design](docs/BLoTv3.md) for its behavior and outstanding qualification limits.

The positive excessive-actuation diagnostic keeps the inherited `2 × ACCEL_MAX` rule: it now detects sustained calibrated/measured acceleration above 8.0 m/s², as in combo. This is distinct from the actual 4.0 m/s² Hyundai command/safety ceiling; the negative threshold, measurement-consistency gate and debounce are unchanged.

Daily's BLoTv3 adaptations include shared primary/secondary-lead landing urgency, a bounded missing-lead urgency
window, and model/radar freshness gates for lead forecasts. They do not add a retained radar obstacle or autonomous
braking authority. Conditional model stops are not a traffic-light detector. See the daily-integration section of
the design document; historical source-branch replay notes are not qualification of this composition.

Conditional Experimental Mode keeps Experimental while the model's plan still sheds most of the car's speed, wherever the
plan ends (a plan that dips toward the line and relaunches no longer reads as an open road), and returns to Chill on the
model's go: 0.5 s of open road with the plan no longer slowing, at rest or below 1 m/s, or with the model asking for
0.5 m/s² or more. An exit taken at rest stays revocable by the model's stop until the car moves off. Entry is unchanged
(BLoTv3 D41; not field validated).

## Lane-centering toggle

**Settings -> Dev -> Lane Centering** is available in both UI layouts. It defaults **on**, persists across restarts, and can be switched without restarting or disengaging. A background reader polls the setting outside the real-time steering loop. Turning it off resets the extra correction; ordinary curvature limits still shape the delivered command. It changes actual steering, not the displayed model path. Have a passenger operate it while moving, or change it while stopped.

Centering fades for driver steering, either blinker, or a non-off lane-change state, and stands down after a driver push in the direction the model leans, as described above. It resets when lateral control is inactive or the lateral-maneuver tool owns the target. The combined signal/start/recenter path has host regression coverage, not physical closed-loop validation. Watch for unexpected weave, curve tracking, or reduced obstacle clearance; no road-validation claim is made here.

## Curve and passing composition (not field validated)

- **Curve-speed limiter:** the current model-path policy anticipates bends and supplies a torque-aware coast/brake candidate. It competes with cruise, lead braking, the model and committed stops; it does not change the driver's set speed or replace steering. Live torque calibration is optional, with the source fallback retained. Its decision is logged as `curvePolicyState`; the existing BLoTv3 `stop` source and SOL telemetry retain their schema IDs. See [curve policy design](docs/ModelCurveSpeedLimit.md).
- **Passing acceleration and relaxed gaps:** a starting lane change with a clear target lane and set-speed headroom can temporarily reduce the primary lead's nominal following time to **0.6 s**, allowing the existing cruise acceleration to take over. This is not an extra acceleration boost or a global personality-gap reduction. The relaxation uses the source's nominal **2.5 s** window and cancels for the source's lead-braking, handover, driver-back-out and reset conditions. BLoTv3's additional protective gap padding remains additive and the stopped gap stays 7 m. Passing uses the same 4.0 m/s² platform ceiling, not an additional increase.
- **Target-lane radar planning:** front radar tracks, lane geometry and the appropriate blind-spot indication gate the longitudinal passing behavior. This requires an eligible radar/BSM vehicle, healthy inputs, and full forward-ordered model/lane polylines. Unusable or non-forward geometry disables the passing policy without removing ordinary lead following or curve control. It is not an independent lane-change safety guarantee and does not add a target-lane radar veto to the lateral nudgeless state machine.
- **Lead release (explicitly selected):** while the source's path, clearance and restoration conditions permit, selected old-lane leads are removed from the MPC planning view and its FCW calculation. They remain in the original `radarState` and `hasLead` publication. Released slots cannot return as model-lead forecasts. Stopped/braking or unsafe-to-restore leads are retained/restored according to the source gates. Release is not time-limited like the shorter gap: it can continue through the starting phase until a restore/end condition. Model braking and committed-stop authority are not lifted by passing.

Short gaps reduce reaction margin. Lead-release mistakes can suppress braking and FCW for a real vehicle; radar/BSM and model lane placement have blind spots and cannot prove the target lane safe. The curve policy can request inappropriate slowing from bad geometry or authority estimates. These are local experimental changes, not road-qualified behavior.

Intentional lead restoration drops only the MPC's obsolete free-run change-cost reference through re-anchoring. Jerk and safety costs remain; ordinary radar dropouts keep their existing behavior. The release guard does not assume current braking will continue after the lead is released.

## Green-light launch assistance (in progress; not field validated)

Better Green Lights uses the model's planned path and motion, **not an independent traffic-light detector**. In effective Experimental mode, a sustained open path above 20 m can clear the model's lingering stop bit while held; path collapse below 10 m, mode/health/reset gates or speed above 2 m/s cancel anticipation. A credible moving plan is time-shifted through its initial hesitation to provide a launch-assist acceleration floor, capped at 1.5 m/s² and faded out by 2 m/s.

This shapes the model candidate only. BLoTv3's committed stops, lead/MPC braking, curve restriction, cruise/force-deceleration target and downstream stopping/vehicle safety remain separate constraints. It cannot declare a light green or the road safe. A false open-path prediction can invite an early launch; driver supervision and immediate braking remain necessary. Malformed/stale model input cannot grant or retain this added launch authority.

The import's monotone-path health check rejected the model's normal path jitter at rest (points stepping back by millimetres to centimetres), so a held green never armed the launch assist. Backward steps are now tolerated up to 5 m/s over the points' time spacing; logged paths below the 2 m/s disarm speed stay under 2.5 m/s, and a reversed path is still rejected.

## Custom comma 3X/Chestnut integration (in progress; not device validated)

- Warp the camera frames on the device GPU, transferring the smaller warped input to the big-model device instead of both raw camera frames. The ordinary same-device path remains supported.
- Preserve Chestnut enumeration before bounded PCIe-link polling and model loading. An already-up link is not disturbed by a redundant power request.
- A still-running loader at the deadline cannot share a process with fallback. Modeld synchronously records `ChestnutLoadTimedOut` and terminates the process; only its dedicated timeout exit permits a manager restart. The replacement selects the small model, and the latch prevents another big-model attempt until manager startup clears it. Ordinary crashes retain upstream no-auto-restart behavior. If the timeout latch cannot be persisted, modeld fails closed without entering a retry loop.
- Only the timeout-governed big-model load uses serial tinygrad compilation (`PARALLEL=0`), preventing its spawned compiler pool from outliving a timed-out loader. Normal compilation settings are restored when that load completes or returns an ordinary error. Cold-start timing under this policy remains a device-validation gate.
- A big-model load that fails without timing out (loader exception, Chestnut not enumerating, PCIe link not up) retries once through a ~1 s onroad cycle (upstream #39086): the first failure sets `ChestnutRetry` and requests the cycle without writing `ChestnutActive`, so **Big Model Failed** is not raised for it; a failure on the retry boot reports the failure and runs the small model. `ChestnutRetry` clears on ignition-on and manager start. The timeout latch above is separate and never retries.
- Add a permanent **Big Model Loading** alert alongside the existing no-entry alert, and warm the standby small model before realtime operation when the big model is selected.
- Check for a materialized big-model pickle rather than requiring the unused big-model warp pickles; this rejects an unfetched LFS pointer but is not a checksum or inference-validation test.

**Big model pinned to Cinque v3 (owner's call, 2026-10-18: it drove better than Mountain Dew).** `big_driving_tinygrad.pkl` is upstream's Cinque v3 pickle (#38932, checkpoint `f78ed37d`, LFS sha256 `4937fc6f…`), the same one `combo` runs. A tinygrad pickle only loads under the tinygrad it was pickled with, so the `tinygrad_repo` pin stays at `9d0446a4b` and `input_view` keeps that pin's `Buffer.view(size, dtype, offset)` call; daily's previous pin `d3f09c9bb` (#39054) raises `TypeError` on this pickle (verified on a CPU unpickle). Upstream's drop-chestnut-on-fallback fix (#39044), written for pin `9d0446a4b` and reverted only as part of the bump, returns with it. Upstream model releases after Cinque v3 (ResAction #39037, Mountain Dew #39045/#39047, MD v3 #39087) and the tinygrad bumps that re-pickled them (#39054, #39085) are deliberately not taken: an upstream import must leave the tinygrad pin and the three `big_driving_*_tinygrad.pkl` pointers alone unless the owner picks a new model. Host tests and CPU stand-ins cannot establish QCOM/USB+AMD compatibility, inference timing or dock-disconnect recovery. Real comma 3X/Chestnut startup, frame budget and offroad fallback checks remain unperformed; no firmware was flashed or device accessed for this import.

## Deliberately excluded

BLaTv2 and the third model-only cut-in obstacle are not imported. Upstream AccelBoost is retained; the Mountain Dew big model and the tinygrad bump that shipped with it are not (see the Chestnut section). The opendbc pin advances for the selected 4.0 m/s² ceiling; existing SOL and torque safety behavior is retained, and unrelated dependency pins are unchanged. The upstream radar publication retains the Smooth Stops unconfirmed low-speed gate; lane-change filtering is a separate planner-local view. MPC collision-warning prediction follows the accepted BLoTv3 trajectory for non-released leads. The accepted upstream type-check diagnostics are not fixed by these imports. All additions remain local pending separate device and owner field acceptance.

## Import provenance

- Better Green Lights: `cec0e0405a4dade816659966de2dc1df4f991ef0`, with `combo` at `907f3965fb92b1572f6d37cb6cf2834157b9de10` as the composed reset/landing reference, adapted to daily's current planner owners and input-health gates.
- Custom comma 3X/Chestnut: `08584ed71ee6144cbe9af937e2bf7857595355e4`, preserving upstream enumeration ordering; the big-model and tinygrad pin were changed afterwards (next item).
- Big model pin: Cinque v3 pickles (`big_driving_tinygrad.pkl` and both `big_driving_warp_*` pointers) and tinygrad `9d0446a4b` from `combo` at `907f3965fb92b1572f6d37cb6cf2834157b9de10` (upstream's state before #39037), plus upstream `a0d47bcd08d9c359ccc7bd73145ed90ade2255ab` (#39044) re-placed around daily's loader. Owner ruling 2026-10-18.
- Big-model load retry: upstream `6939d172e85e3b91f16d90cc8de72e8f938cc196` (#39086), placed after daily's timeout latch with the drop-chestnut fallback kept unconditional.
- Ignored-uploads offroad alert: upstream `35e78a9d9518855ec0dacec919e719624a28131f` (#39104), applied unchanged.
- Initial Max Speed setting: upstream `24c26487f42b1d92cf96e3e23d686208e35e74e3` (#39081), applied unchanged; daily's cruise-button tests now pass an initial speed in km/h.
- Following Distance label: upstream `41acce17fa226cb40af8f3e17f64259784731b01` (#39100), applied unchanged; translations were not regenerated, so the renamed label shows in English in other languages until upstream's catalog update is taken.
- Acceleration ceiling: the 4.0 m/s² generic/Hyundai host and native Hyundai limit from combo's opendbc reference `97640dca1bb21174de16503ad53a18bbbb3efa8e`; no wholesale dependency replacement or unrelated safety-limit import.

- Current curve-speed policy: `e976ab76aa4bd394ff2d5baeb0591623eb893f92`, composed with BLoTv3 arbitration; its unrelated `teleoprtc_repo` pin change is excluded.
- Passing gap/acceleration, target-lane checks and lead release: `ac85e99c3049f4a88c376deae94ba46af99b50d5`, composed with BLoTv3's existing MPC gap API and predictive lead ownership. These are feature-delta imports, not whole-source-branch ancestry merges.

- ESP-active debounce: `e9fd5e680cc0e80934295fb210483e8ceadc2d70`.
- Hot-swap Experimental: `a91715adbf7605dba9c934e9de545fa99a0c0b40`, adapted for daily's manual/CEM ownership, asynchronous settings updates and the Hyundai CAN-FD input limitation.
- Swapped cruise-speed steps: `091691fd8873e1c7b06b2503f57f415544da8ffc`.
- Smooth Stops radar filter: age gate `97da369c4bded416fb5dc68c6e09a62b66140a55` and distance floor `61cc462bbc551c5d0bb4bca146301e6d4de612fe`, now carried on BLoTv3 at afa3adf7f1c2980d545e2d9a5325f68183d34670; not the historical LongControl implementation.

- Full longitudinal feature: [BLoTv3 at 926ccb7a3](https://github.com/SpysyWeeb/Spysypilot/tree/926ccb7a3f24ab49d47b64fde2cbcc98ff02e14f), reconciled with daily’s newer upstream planner and SOL. Local verification is not device or field acceptance.
- Conditional Experimental Mode slowing hold and go exit: BLoTv3 `5dac58958257cb6a14b432a07e206b4a9f73881f` (D41), applied unchanged; `conditional_experimental_mode.py`, `stop_helpers.py` and their tests match BLoTv3's, and the design document carries D40/D41.

- Nudgeless production helper and source-owned tests: [lane-changes at ac85e99c](https://github.com/SpysyWeeb/Spysypilot/tree/ac85e99c3049f4a88c376deae94ba46af99b50d5), including the early-manual-start retry fix. Its lateral behavior is unchanged by the subsequent longitudinal passing import.
- Complete centering feature and both Dev toggles: [lane-centering at a3f3b942](https://github.com/SpysyWeeb/Spysypilot/tree/a3f3b94250611608dc39b9357e1364ea1b091960). Test setup is adapted to daily's required SOL status; production behavior is unchanged.
- Heading-aware fade and driver push: [lane-centering at 3f92d7cb9](https://github.com/SpysyWeeb/Spysypilot/tree/3f92d7cb92e2548e51a6fc6d087000338493a1f4), cherry-picked as feature-delta commits without the branch's newer upstream merges.

---

<div align="center" style="text-align: center;">

<h1>openpilot</h1>

<p>
  <b>openpilot is an operating system for robotics.</b>
  <br>
  Currently, it upgrades the driver assistance system in 300+ supported cars.
</p>

<h3>
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://docs.comma.ai/contributing/roadmap/">Roadmap</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Community</a>
  <span> · </span>
  <a href="https://comma.ai/shop">Try it on a comma four</a>
</h3>

Quick start: `bash <(curl -fsSL openpilot.comma.ai)`

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63"></a></td>
  </tr>
</table>


Using openpilot in a car
------

To use openpilot in a car, you need four things:
1. **Supported Device:** a comma four, available at [comma.ai/shop/comma-four](https://www.comma.ai/shop/comma-four).
2. **Software:** The setup procedure for the comma four allows users to enter a URL for custom software. Use the URL `openpilot.comma.ai` to install the release version.
3. **Supported Car:** Ensure that you have one of [the 300+ supported cars](docs/CARS.md).
4. **Car Harness:** You will also need a [car harness](https://comma.ai/shop/car-harness) to connect your comma four to your car.

We have detailed instructions for [how to install the harness and device in a car](https://comma.ai/setup). Note that it's possible to run openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), although it's not plug-and-play.


### Branches

Running `master` and other branches directly is supported, but it's recommended to run one of the following prebuilt branches:

| comma four branch      | comma 3X branch        | URL                                    | description                                                                         |
|------------------------|------------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release-mici`         | `release-tizi`         | openpilot.comma.ai                     | This is openpilot's release branch.                                                 |
| `release-mici-staging` | `release-tizi-staging` | openpilot-test.comma.ai                | This is the staging branch for releases. Use it to get new releases slightly early. |
| `nightly`              | `nightly`              | openpilot-nightly.comma.ai             | This is the bleeding edge development branch. Do not expect this to be stable.      |
| `nightly-dev`          | `nightly-dev`          | installer.comma.ai/commaai/nightly-dev | Same as nightly, but includes experimental development features for some cars.      |

For [chestnut](https://comma.ai/shop/chestnut), use the following installer URLs:

| branch                       | URL                                                        | description                                                                         |
|------------------------------|------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `release-chestnut`           | installer.comma.ai/commaai/release-chestnut                | This is openpilot's release branch.                                                 |
| `release-chestnut-staging`   | installer.comma.ai/commaai/release-chestnut-staging        | This is the staging branch for releases. Use it to get new releases slightly early. |
| `nightly-chestnut`           | installer.comma.ai/commaai/nightly-chestnut                | This is the bleeding edge development branch. Do not expect this to be stable.      |
| `nightly-chestnut-dev`       | installer.comma.ai/commaai/nightly-chestnut-dev            | Same as nightly, but includes experimental development features for some cars.      |

To start developing openpilot
------

openpilot is developed by [comma](https://comma.ai/) and by users like you. We welcome both pull requests and issues on [GitHub](http://github.com/commaai/openpilot).

* Join the [community Discord](https://discord.comma.ai)
* Check out [the contributing docs](docs/CONTRIBUTING.md)
* Check out the [openpilot tools](openpilot/tools/)
* Code documentation lives at https://docs.comma.ai
* Information about running openpilot lives on the [community wiki](https://github.com/commaai/openpilot/wiki)

Want to get paid to work on openpilot? [comma is hiring](https://comma.ai/jobs#open-positions) and offers lots of [bounties](https://comma.ai/bounties) for external contributors.

Safety and Testing
----

* openpilot observes [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md) for more details.
* openpilot has software-in-the-loop [tests](.github/workflows/tests.yaml) that run on every commit.
* The code enforcing the safety model lives in panda and is written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
* panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
* Internally, we have a hardware-in-the-loop Jenkins test suite that builds and unit tests the various processes.
* panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
* We run the latest openpilot in a testing closet containing 10 comma devices continuously replaying routes.

<details>
<summary>MIT Licensed</summary>

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and comma Account</summary>

By default, openpilot uploads driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software, and users can disable data collection if they wish.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
