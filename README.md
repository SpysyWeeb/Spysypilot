# Model Curve Speed Limit — In Progress

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot).

Status: **in progress pending field testing**. This feature must not be marked
complete until the owner has tested it on-device.

**2026-11-02 — known issue: a bend banked against the turn is driven 15 mph under the owner's pace.** On route 0x1a4 (daily `a4acc083d`, which runs this policy), a gentle left bend (R ≈ 130 m) whose road falls away to the right (live roll +3.5…+5.7°) was braked from 34.7 to 19.4 mph. The owner drives the same bend at 34.4–35 mph (route 0x1b0, openpilot steering, speed by foot): 1.74–1.87 m/s² of lateral, torque 1.0 for ~1.5 s, tracking within 0.1 m/s². Both passes were replayed from their rlogs (the replay reproduces the logged `curvePolicyState` on 661 of 662 frames) and re-measured by independent adversarial checks. This branch's module makes identical decisions on both passes, so all of the following applies here:

- **Anticipation: ~11 of the 15 mph.** The strictest node is always the last one inside `ROLL_HORIZON_S` (~30 m), where the live roll is charged in full: 0.66–0.99 off a 1.73–1.92 m/s² margin. The next node, 4.5 m farther with the same curvature, gets no bank and prices 7 mph faster (29.0 vs 36.4 mph). Taken one term at a time: the bank costs 8–11 mph; the `AUTHORITY_HELD_LATERAL` floor, blended out at 15.5 m/s by `AUTHORITY_HELD_SPEED` (15–20 m/s), 7–7.5; the authority factor (2.2–2.5 priced, 2.81 measured at full torque on the owner's pass) 4–5; `TORQUE_BUDGET` 0.90 against the 1.0 the owner used, 2–3. Forced on over the owner's own pass, the policy asks for −2.0 m/s² (limit 26 mph).
- **Brake regime: ~4 mph.** It entered on understeer that crossed `E_TRACK` (0.30) by 0.003–0.048, and 0.06–0.08 of each qualifying frame was the policy's own braking. The torque controller's desired lateral uses v² from 0.21 s earlier while the actual uses the current v², which is about 0.05 m/s² of apparent understeer per m/s² of deceleration. Without that term no frame reaches 0.30. The line loss was real but small: 0.2–0.3 m wider at the apex, still inside the lane.
- **"Pinned" is torque only.** The torque controller's saturated flag never set on either pass (the driver-torque trim keeps the safety limit active), so the brake regime's pin test reduces to torque ≥ `T_PIN`.
- **Open question:** at full torque the steering delivered 2.51 m/s² of its own lateral on the owner's pass but 1.85 on 0x1a4. Braking was suspected, but a smaller gap was already there before braking began, so the cause is not established.

This is not a regression: daily's module before 2026-11-01, after it, and this branch's all make the same decisions here. On daily, the 2026-11-01 bank continuation also lowers far limits on a bend banked against the turn (by up to 10.8 mph here), the opposite of its intent as a credit; it did not bind on this pass. Status: open, not fixed on this branch or on daily.

2026-09-15: audit fixes — the brake regime prices the restore no higher than the authority the steering delivers at its torque limit, the 2.5 m/s² demonstrated hold applies to the Palisade only, brake recovery must persist 0.5 s (`BRAKE_EXIT_S`), the maneuver plant saturates both curve directions (mirrored maneuver added), and the decision is logged as `curvePolicyState`. Awaiting the owner's drive.

2026-09-05: route 0x54's late lift — the authority calibration samples only at torque ≥ 0.70 (`AUTHORITY_MIN_TORQUE`, the ratio is not constant in torque and the plan is aimed at the 0.90 budget), and a path node beyond the roll horizon whose shortfall a bank could explain (within 25 % of the speed) asks for a moderate deceleration at most (`FAR_NODE_DECEL_MAX` 1.0 m/s²). Replay-gated on routes 0x54, 0x4c, 0x4d; awaiting the owner's drive.

2026-09-04: the same bend twice (routes 0x4c/0x4d) — far path nodes no longer get the car's live roll (`ROLL_HORIZON_S` 2 s), the authority calibration learns only from real corners at speed with the bank taken out of the measurement, and the hold releases as a `J_UP` ramp instead of a step. Replay-gated on both passes; awaiting the owner's drive. Details: docs/ModelCurveSpeedLimit.md.

2026-09-02: a lift now ends in a hold — after the coast or brake regime releases, the candidate holds zero until the bend reads open (`BEND_OPEN_A_LAT` 1.0 m/s² for 1 s) instead of handing the plan back to the accelerating candidate (route 0x33: 16 → 4 source handoffs). Replay-gated, plant-gated, awaiting the owner's drive. Details: docs/ModelCurveSpeedLimit.md.

Rebuilt 2026-08-29 as a curve longitudinal *policy*: the model path's curves become one
acceleration candidate in the planner's arbitration (anticipation from a per-node torque
budget whose steering authority is calibrated online), and the measured steering state
decides in-curve behaviour — heavy torque lifts off the throttle, a pinned and understeering
car brakes to the speed that restores margin. The previous speed cap, its 0.2 m/s² release
and the 90 % predicted-torque veto are gone. See
[`docs/ModelCurveSpeedLimit.md`](docs/ModelCurveSpeedLimit.md).

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
