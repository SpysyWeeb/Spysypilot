# Spysypilot daily

A minimal-diff personal branch on the official Mountain Dew master baseline. Experimental controller changes remain subject to owner/device/field validation; local tests are not a driving-safety qualification.

## Selected features

- **Torque + SOL:** default classic-Hyundai 409 / +4 / -7 envelope, retaining existing lower-limit/CAN-FD exceptions and paired safety. SOL provides independent lateral operation, fault/freshness and monitoring integration. Driver steering caps an available SOL torque request at +/-0.5; smaller ordinary requests stay smaller, and downstream limits still apply.
- **Nudgeless lane changes:** a turn signal can initiate the existing lane-change maneuver without a steering nudge, retaining speed and blind-spot gates. The first start, automatic or manual, consumes that signal event's automatic allowance. Holding the signal does not grant another automatic attempt; another manual nudge is still permitted, and cycling the signal rearms automatic operation. Braking cancels the automatic allowance for that signal event. This does not add release-and-nudge edge detection or prove a lane line was crossed.
- **Lane centering:** the feature branch's correction, filter, confidence/geometry checks, one-line width hold, and live toggle are imported without changing their behavior. The correction feeds the existing curvature limits and controller, then SOL's downstream driver-input cap. It also operates during SOL-only lateral activation.

- **ESP-active debounce:** the common car-event path raises `espActive` after 50 consecutive active samples (nominally 0.5 s). Any inactive sample resets it. This delays genuine interventions as well as nuisance pulses; it is not a bump detector. `espDisabled`, other faults, the existing downstream soft-disable policy, and vehicle/panda safety behavior are unchanged.
- **Hot-swap Experimental:** hold the distance button for 0.5 s to toggle the saved manual Experimental preference once per hold; a short tap still cycles personality. Requires openpilot longitudinal control. The hold action is disabled on Hyundai CAN-FD, whose edge-only button stream can pause long enough to misclassify a tap; release still cycles personality there. The classic-CAN Palisade is eligible. Effective mode remains manual OR Conditional Experimental Mode, so manual-off does not veto an automatic CEM request.
- **Swapped cruise-speed steps:** for openpilot-owned set speed, short taps move to the next five-unit grid point and held presses repeat one-unit steps. Metric/imperial conversion, speed bounds, engagement, gas-override and standstill-resume guards remain. Vehicle-owned cruise speed is unchanged.
- **Smooth Stops radar filter:** the current branch adds an age/range gate only to unconfirmed low-speed lead overrides: 20 consecutive radar updates and a distance above `max(0.75 m, 0.6 s × ego speed)`, retaining the existing below-4 m/s, lateral and 25 m gates. Model-confirmed leads bypass this extra gate; a missing radar cycle restarts track age. This can also delay or reject a real newly appearing, flickering or close unconfirmed object. BLoTv3's landing and the existing LongControl remain unchanged.

- **BLoTv3 longitudinal (in progress; not field validated):** full model-lead forecasting, following-response supervision, cruise comfort/tuning, Conditional Experimental Mode, committed model stops, lead hold/departure handling, planner handoffs and final stop landing. Newer upstream AccelBoost is retained. The platform acceleration ceiling and paired safety pins are unchanged; the nominal stopped-lead distance is 7 m and aggressive base following time is 1.0 s. This replaces stock longitudinal planning, not radar selection or steering. See [BLoTv3 design](docs/BLoTv3.md) for its behavior and outstanding qualification limits.

Daily's BLoTv3 adaptations include shared primary/secondary-lead landing urgency, a bounded missing-lead urgency
window, and model/radar freshness gates for lead forecasts. They do not add a retained radar obstacle or autonomous
braking authority. Conditional model stops are not a traffic-light detector. See the daily-integration section of
the design document; historical source-branch replay notes are not qualification of this composition.

## Lane-centering toggle

**Settings -> Dev -> Lane Centering** is available in both UI layouts. It defaults **on**, persists across restarts, and can be switched without restarting or disengaging. A background reader polls the setting outside the real-time steering loop. Turning it off resets the extra correction; ordinary curvature limits still shape the delivered command. It changes actual steering, not the displayed model path. Have a passenger operate it while moving, or change it while stopped.

Centering fades for driver steering, either blinker, or a non-off lane-change state. It resets when lateral control is inactive or the lateral-maneuver tool owns the target. The combined signal/start/recenter path has host regression coverage, not physical closed-loop validation. Watch for unexpected weave, curve tracking, or reduced obstacle clearance; no road-validation claim is made here.

## Curve and passing composition (not field validated)

- **Curve-speed limiter:** the current model-path policy anticipates bends and supplies a torque-aware coast/brake candidate. It competes with cruise, lead braking, the model and committed stops; it does not change the driver's set speed or replace steering. Live torque calibration is optional, with the source fallback retained. Its decision is logged as `curvePolicyState`; the existing BLoTv3 `stop` source and SOL telemetry retain their schema IDs. See [curve policy design](docs/ModelCurveSpeedLimit.md).
- **Passing acceleration and relaxed gaps:** a starting lane change with a clear target lane and set-speed headroom can temporarily reduce the primary lead's nominal following time to **0.6 s**, allowing the existing cruise acceleration to take over. This is not an extra acceleration boost or a global personality-gap reduction. The relaxation uses the source's nominal **2.5 s** window and cancels for the source's lead-braking, handover, driver-back-out and reset conditions. BLoTv3's additional protective gap padding remains additive; the stopped gap and platform acceleration ceiling are unchanged.
- **Target-lane radar planning:** front radar tracks, lane geometry and the appropriate blind-spot indication gate the longitudinal passing behavior. This requires an eligible radar/BSM vehicle, healthy inputs, and full forward-ordered model/lane polylines. Unusable or non-forward geometry disables the passing policy without removing ordinary lead following or curve control. It is not an independent lane-change safety guarantee and does not add a target-lane radar veto to the lateral nudgeless state machine.
- **Lead release (explicitly selected):** while the source's path, clearance and restoration conditions permit, selected old-lane leads are removed from the MPC planning view and its FCW calculation. They remain in the original `radarState` and `hasLead` publication. Released slots cannot return as model-lead forecasts. Stopped/braking or unsafe-to-restore leads are retained/restored according to the source gates. Release is not time-limited like the shorter gap: it can continue through the starting phase until a restore/end condition. Model braking and committed-stop authority are not lifted by passing.

Short gaps reduce reaction margin. Lead-release mistakes can suppress braking and FCW for a real vehicle; radar/BSM and model lane placement have blind spots and cannot prove the target lane safe. The curve policy can request inappropriate slowing from bad geometry or authority estimates. These are local experimental changes, not road-qualified behavior.

Intentional lead restoration drops only the MPC's obsolete free-run change-cost reference through re-anchoring. Jerk and safety costs remain; ordinary radar dropouts keep their existing behavior. The release guard does not assume current braking will continue after the lead is released.

## Deliberately excluded

BLaTv2 and the third model-only cut-in obstacle are not imported. Upstream AccelBoost, models, and the existing paired opendbc/panda pins are retained. The upstream radar publication retains the Smooth Stops unconfirmed low-speed gate; lane-change filtering is a separate planner-local view. MPC collision-warning prediction follows the accepted BLoTv3 trajectory for non-released leads. The accepted upstream type-check diagnostics are not fixed by these imports. All additions remain local pending separate device and owner field acceptance.

## Import provenance

- Current curve-speed policy: `e976ab76aa4bd394ff2d5baeb0591623eb893f92`, composed with BLoTv3 arbitration; its unrelated `teleoprtc_repo` pin change is excluded.
- Passing gap/acceleration, target-lane checks and lead release: `ac85e99c3049f4a88c376deae94ba46af99b50d5`, composed with BLoTv3's existing MPC gap API and predictive lead ownership. These are feature-delta imports, not whole-source-branch ancestry merges.

- ESP-active debounce: `e9fd5e680cc0e80934295fb210483e8ceadc2d70`.
- Hot-swap Experimental: `a91715adbf7605dba9c934e9de545fa99a0c0b40`, adapted for daily's manual/CEM ownership, asynchronous settings updates and the Hyundai CAN-FD input limitation.
- Swapped cruise-speed steps: `091691fd8873e1c7b06b2503f57f415544da8ffc`.
- Smooth Stops radar filter: `07705fad18ee7b6024d19e37edb3ccd0dcefe542`; not the historical LongControl implementation.

- Full longitudinal feature: [BLoTv3 at 926ccb7a3](https://github.com/SpysyWeeb/Spysypilot/tree/926ccb7a3f24ab49d47b64fde2cbcc98ff02e14f), reconciled with daily’s newer upstream planner and SOL. Local verification is not device or field acceptance.

- Nudgeless production helper and source-owned tests: [lane-changes at ac85e99c](https://github.com/SpysyWeeb/Spysypilot/tree/ac85e99c3049f4a88c376deae94ba46af99b50d5), including the early-manual-start retry fix. Its lateral behavior is unchanged by the subsequent longitudinal passing import.
- Complete centering feature and both Dev toggles: [lane-centering at a3f3b942](https://github.com/SpysyWeeb/Spysypilot/tree/a3f3b94250611608dc39b9357e1364ea1b091960). Test setup is adapted to daily's required SOL status; production behavior is unchanged.

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
