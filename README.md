# BLoTv2 — Better Longitudinal Tune v2

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot),
created from the current untouched `stock` tip.

## Status

**In progress — not field validated.**

BLoTv2 is a ground-up longitudinal control iteration that combines an
improved Smooth Stops final-approach controller with an improved
longitudinal planner/MPC policy. Its joint product target is:

- **Smooth:** continuous acceleration and jerk through following, braking,
  final approach, hold, and release.
- **Swift:** prompt response to real lead and model intent changes, without
  stale state or hidden delay.
- **Strong:** proportional use of the safe acceleration envelope when the
  situation requires it, without weakening stock emergency braking or
  platform safety limits.

The current test revision retains up to `4.0 m/s²` at launch. Field feedback
from route `000000d2--a62f0c1831` now fades that extra launch authority into
openpilot's stock acceleration gate by `10 m/s` (about `22 mph`) and follows
the stock ceiling above it. The reaction-time jerk tune remains unchanged.
Requested authority remains clamped to the deployed opendbc limit.

Route `000000d9--6040563d1d` showed that the stock ceiling alone still makes a
small highway set-speed correction use all available acceleration or the full
`-1.2 m/s²` cruise deceleration limit. The current in-progress tune adds a
lead-free Chill cruise comfort response: above `15 m/s`, a `5 mph` error asks
for about `0.40 m/s²`, tapers continuously as the error closes, and uses the
pitch-compensated coast estimate during speed reductions. It blends in from
`8–15 m/s`, so low-speed launch response is retained, while larger errors can
still reach the existing acceleration and braking envelope. Experimental
mode, forced deceleration, radar-invalid or lead-following operation, and an
active model curve-speed limit bypass this shaping. The jerk schedule remains
unchanged. This is route-replay tuning and still requires owner field testing.

Conditional Experimental Mode is **in progress**. Its target behavior is to
keep BLoTv2 in Chill mode during ordinary driving, hand the existing
longitudinal planner to Experimental mode for a confirmed, lead-free model
stop prediction, hold that handoff through the stop, and return to Chill after
a stable release or driver override. This branch has no standalone traffic
light or stop-sign classifier, so the detector uses the model's existing
`action.shouldStop`, predicted position/velocity/orientation, and action
signals. A route-derived high-speed tier now recognizes a straight, lead-free
near-stop trajectory before the terminal prediction reaches zero; weaker
evidence only precharges its existing temporal filter. Replay of route
`000000d7--cc6308b4d0` moves the first handoff from `39.1 mph` to `42.2 mph`
without admitting its highway-exit slowdown or brief radar-dropout window.
This is counterfactual replay evidence, not field validation. The feature
publishes only through `selfdriveState.experimentalMode`; it adds no UI or
user-facing Params and does not own target speed or braking.
The implementation and signal mapping are documented in
[`docs/BLoTv2.md`](docs/BLoTv2.md#conditional-experimental-mode).

Each physical decision has one owner. The planner/MPC owns trajectory
necessity, Smooth Stops owns only the final low-speed approach and hold
handoff, and the stock long controller and platform safety layers retain
their existing command-envelope responsibilities.

The design, exact mechanisms, known risks, and field plan are documented in
[`docs/BLoTv2.md`](docs/BLoTv2.md). Promotion gates are tracked in
[`docs/BLoTv2_ACCEPTANCE.md`](docs/BLoTv2_ACCEPTANCE.md).

This status remains **in progress** until the owner completes a field test and
explicitly authorizes changing it.

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
