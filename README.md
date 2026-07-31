# BLaTv2 — modular adaptive lateral control

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot),
created from the current untouched `stock` tip.

## Status

**In progress — ground-up replacement. Not field eligible.**

This branch replaces the previous BLaTv2 controller architecture. Git history
and useful test infrastructure remain available for audit, but no previous
BLaTv2 controller mechanism is part of the new active design by default.
Until the complete modular candidate passes replay, safety, timing, and
portability gates, the stock openpilot torque controller remains the only
active controller. The candidate is evaluated in offline replay and harness
tests; the field manager does not launch a BLaTv2 shadow process onroad.

The product target is **Smooth. Swift. Strong.**

- **Smooth:** the requested wheel motion is continuous and intentional, without
  jitter, ping-pong, or modules correcting one another.
- **Swift:** useful torque is requested quickly enough to meet the model's
  authored path timing. The path is not moved earlier to conceal controller
  delay.
- **Strong:** the controller uses the vehicle's available authority when the
  model path requires it and does not accept persistent tracking error merely
  to keep the torque trace quiet.

The architecture, module contracts, learning policy, opendbc boundary, and
acceptance sequence are documented in
[`docs/BLATV2_MODULAR.md`](docs/BLATV2_MODULAR.md) and
[`docs/BLATV2_ACCEPTANCE.md`](docs/BLATV2_ACCEPTANCE.md).

No BLaTv1 controller code or `HyundaiLowSpeedTorqueDamping` is inherited.
Vehicle-specific magnitude/rate limits come from the active opendbc
`CarControllerParams`; the controller contains no Palisade limit literals.
opendbc may enforce platform command limits and panda safety, but it may not
damp, boost, filter, or otherwise reinterpret the controller's normalized
request.

### What is built

The replacement is split into independently testable owners:

- a model-time/intent adapter and stateless scalar-anchored future reference;
- one measured rack mapping and forward/inverse physical plant;
- one computed-torque core;
- one exact opendbc command envelope and invalid-output guard;
- an offline measured-response shadow and an offroad full-rlog importer for
  durable speed-local learning;
- an offroad-only artifact, feedback, promotion, and rollback lifecycle.

The previous **LQI** controller—“Linear Quadratic Integral,” a state-feedback
controller with an integral error state—is retired. It is not the controller
being built here. The replacement core directly computes the torque required
by the authored rack position, rate, and acceleration using one learned
physical profile, so a tuning module does not have to correct another tuning
module.

The first combo build contains no approved modular profile. It therefore
drives with the exact current stock torque-controller algorithm and launches
no dedicated BLaTv2 process onroad. Normal loggerd full rlogs retain the
measured services needed for later learning. After logger closure, the
offroad importer independently replays the complete full rlog before any
evidence can become durable. On opendbc's validated Palisade/Telluride
platform, that stock request passes through the
platform-selected **409/4/7** opendbc/panda envelope. Other cars keep their
own stock limits and remain stock-controlled unless their opendbc port
explicitly validates the complete modular command-envelope and rack-sensor
contract.

## Activation and learning

The first drive on a vehicle is stock-controlled. No BLaTv2 collection or
learning process runs during the drive. After the route is closed,
`blatv2_backfilld` replays its complete full rlog twice and atomically commits
evidence only if both replays and all compatibility checks agree. It is the
sole managed BLaTv2 process and the sole durable learning writer.

The importer can also use older local routes, including routes recorded before
this importer existed, when their complete full rlogs remain on the device and
their exact build/schema, dongle, vehicle, CarParams, controller-envelope,
sensor-resolution, and source-coverage checks pass. Qlogs, incomplete or
open routes, unreviewed builds, incompatible routes, and late-discovered older
routes are not imported. A rejected route does not block a later compatible
one.

A candidate profile becomes eligible only when every required speed region
has enough excitation, independent validation, and bounded uncertainty.
Highway mileage cannot overwrite low-speed knowledge because samples update
only their neighboring speed nodes.

Profiles never change mid-drive. A newly qualified profile is promoted only at
an engagement boundary and can be rolled back at the next disengagement. After
the first active validation drive, the offroad review asks:

> Compared with the previous steering profile, how did steering feel?

The choices are **Better**, **About same**, **Worse**, and **Not sure**.
`Worse` deactivates the provisional profile for the next engagement but keeps
the data for diagnosis. Driver overrides are evidence bookmarks, not automatic
proof that a controller is bad.

Profile revisions are opaque, monotonically increasing evidence generations,
not contiguous release numbers. Identical restored evidence produces the same
revision and hash; any additional accepted clean sample advances the next
qualified candidate. This lets later casual drives refine the profile without
rebasing accumulated physical statistics or letting highway data overwrite
low-speed nodes.

Measured evidence includes reachable, driver-free vehicle-owned torque
boundaries. Full-torque and maximum-slew frames are retained in a separate
speed-local authority stratum rather than being discarded. Slew transients
remain out of the instantaneous equality fit because the rack response is
transport-delayed; settled full-torque motion enters that fit only after the
seed transport delay has elapsed. Driver-limited and physically impossible
transitions remain excluded. This keeps sharp-turn/breakaway evidence without
turning limiter timing or human torque into false plant parameters.
Settled authority rows cannot influence a candidate until their own held-out
validation block exists; incomplete authority evidence stays durable and
deferred.

The home-screen learning display reads rebuildable
`BLaTv2LearningOperationStatus` and `BLaTv2LearningStatus` caches. The
offroad importer owns both. Operation status distinguishes logger
finalization, historical route scanning/replay progress, idle evidence, an
eligible empty state, and fail-closed diagnostics.

Both caches are cleared at manager start and published only after the offroad
owner validates its phase and current vehicle/build authority. They are
informational only: neither is evidence, approval, a profile,
controller-selection, or a safety input, and editing or deleting either one
cannot change steering. The `BLaTv2LifecycleStatus` schema and
`blatv2_profiled` implementation remain available for offline lifecycle
testing, but the stock-only field manager does not launch that process or
publish that cache.

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
