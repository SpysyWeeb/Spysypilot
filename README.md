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
- an offroad full-rlog importer and observable, speed-local inverse-torque
  learner;
- offline-only artifact, feedback, promotion, and rollback test surfaces for a
  future reviewed consumer; none is manager-launched in this milestone.

The current learning milestone replaces the rejected dynamic-rack fit. Normal
driving did not independently identify rack gain and damping, so those
provisional values are no longer learned or allowed to influence learning
identity. Evidence schema 5 instead learns only quantities the logs directly
support at the `0/5/10/15/20/30 m/s` nodes:

- normalized torque per measured lateral acceleration;
- a signed residual lateral-acceleration offset correction;
- moving (kinetic) friction from resolved rack motion; and
- static breakaway torque from the first resolved motion after a measured
  stationary dwell.

The fit is deterministic constrained least squares. Gain, moving friction,
and the excess of static over moving friction are non-negative by physical
construction; the signed offset remains free. This is not a post-fit clamp:
each active constraint is re-solved. If ordinary driving cannot distinguish
extra breakaway, the reported boundary is `static == kinetic` rather than an
invented stiction value.

Stationary settled response, moving response, breakaway events, held-out
validation, and actuator-authority observations remain separate populations.
No partial node set can emit a candidate, and no candidate produced by this
milestone has an approval or activation path.

The previous **LQI** controller—“Linear Quadratic Integral,” a state-feedback
controller with an integral error state—is retired. It is not the controller
being built here. This milestone changes only offline calibration. The exact
stock torque-controller algorithm remains the active bootstrap, and a future
controller must consume one qualified calibration map in place of its stock
map—not add another correction loop around it.

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

**Four-worker backfill expansion is in progress pending its device resource
check.** The offroad importer still has exactly two independent deterministic
replay authorities: the parent and a verification process. Each authority now
has one private route-preparation lane that may decode the next route while its
owner applies the current route, so up to four Python lanes can use the four
comma CPU cores without parallelizing the learner's causal route/frame state.
Each helper writes one versioned, hash-bound, fixed-record scratch spool; it
never sends a route-sized Python object over IPC, never shares prepared data
with the other replay, and never receives durable-writer or Params authority.
Only the parent compares the two results, extends the ledger, and atomically
publishes. One prefetched route per authority bounds memory and scratch usage.
While a prefetched route is being applied, its helper may already be writing
the following route, so the hard scratch bound is two spools per authority:
about 464 MiB total at the one-million-frame safety limit (ordinary routes are
far smaller). Device validation measures the actual high-water mark.
The existing progress UI is deliberately unchanged: it projects the primary
pass, then moves to verification while the independently reconstructed pass
finishes; helper work is accounted in canonical route order and is not exposed
as a third or fourth replay pass. Four lanes improve a multi-route backlog but
cannot accelerate a single newly completed route because there is no next
route to prepare. Worker counts 1 and 2 remain deterministic diagnostic modes;
3 is rejected because it would make the A/A authorities asymmetric.
On the 21-segment `000000b7--a6b3b1f175` reference route, the same desktop
host completed the two passes in 17.594 s serially and 9.717 s with two
workers (1.81x), with identical evidence, manifest, and ledger content. That
is supporting evidence only. The integrated native-extractor/A/A/publication
benchmark over b7, b8, b9, and ca took a 33.754 s four-lane median versus a
42.359 s two-lane median: 20.3% less wall time (1.25x throughput), with exact
evidence, manifest, ledger, provenance, and generation hashes. Four lanes
peaked at about 1.21 GiB process-tree PSS and 26.8 MiB scratch, then left no
children or scratch behind. The comma device has no swap, so elapsed time,
peak memory/scratch I/O, thermals, responsiveness, and identical published
hashes remain the field gate; the task stays in progress until that check.

Its separate display-only progress projection reports the current pass,
route, segment, and whether the route is being read or applied. A cumulative
bar spans both passes without reaching a pass boundary before the prepared
route has actually been ingested. An approximate remaining time appears only
after independent reading and application rates have enough observations;
none of these timing fields enter evidence or determinism comparisons.

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

The learner's current evidence namespace is
`complete_full_rlog_authority_v3`. It starts empty by design. The retired v1
and v2 namespaces and their artifact bytes are never migrated, edited, or
interpreted as schema-5 evidence; compatible retained full rlogs are replayed
from source. Runtime identity for this namespace excludes the retired
provisional rack-gain/damping seed while remaining bound to the detected
vehicle, torque mapping, rack mapping, sensor resolution, and opendbc command
envelope.

This milestone stops at an unapproved, informational candidate file. It does
not promote, activate, or roll back that file, and stock remains selected even
when every node qualifies. The following lifecycle is the separately gated
future consumer contract, not current behavior: a reviewed profile could
change only at an engagement boundary, and after its first active validation
drive the offroad review would ask:

> Compared with the previous steering profile, how did steering feel?

The choices are **Better**, **About same**, **Worse**, and **Not sure**.
Under that future contract, `Worse` would deactivate the provisional profile
for the next engagement while retaining its data for diagnosis. Driver
overrides remain evidence bookmarks, not automatic proof that a controller is
bad.

Profile revisions are opaque, monotonically increasing evidence generations,
not contiguous release numbers. Identical restored evidence produces the same
revision and hash; any additional accepted clean sample advances the next
qualified candidate. This lets later casual drives refine the profile without
rebasing accumulated physical statistics or letting highway data overwrite
low-speed nodes.

Measured evidence includes reachable, driver-free vehicle-owned torque
boundaries. Full-torque and maximum-slew frames are retained in a separate
speed-local authority stratum rather than being discarded. Slew transients
and stationary full-torque rows remain observations, not equality-fit rows;
only settled magnitude-boundary rows with resolved rack motion may join the
fit. Every response frame is paired
causally with the newest recorded torque effective no later than
`response time - seed transport delay`; it is never paired with the convenient
same-frame `carOutput`. `carOutput` itself reports the prior card cycle, so the
effective input clock is the preceding `carOutput` publication. Once aligned,
settled full-torque motion needs one response interval of command-side dwell,
not a second copy of the transport delay. Driver-limited and physically
impossible transitions remain excluded. This keeps sharp-turn/breakaway
evidence without turning limiter timing or human torque into false plant
parameters.

Vehicle steering-rate signals are also normalized before fitting. A natively
signed source keeps its sign. An unsigned magnitude source, including the
Palisade SAS rate, preserves the sensor magnitude and derives direction from
offset-corrected measured steering-angle motion. Quantized plateaus may retain
direction only within one continuous validity epoch. Because the observable
inverse map does not use rack acceleration, a valid measured reversal remains
a moving-response row and direction-coverage event without importing a
quantized acceleration impulse. It cannot manufacture a breakaway event from
stale dwell. Zero motion clears the unsigned-source direction latch; gaps,
disengagement, driver override, standstill, faults, and mapping failures break
cross-frame reversal continuity.
Settled authority rows cannot influence a candidate until their own held-out
validation block exists; incomplete authority evidence stays durable and
deferred.

The home-screen learning display reads rebuildable
`BLaTv2LearningOperationStatus` and `BLaTv2LearningStatus` caches. The
offroad importer owns both. Operation status distinguishes logger
finalization, historical route scanning/replay progress, idle evidence, an
eligible empty state, and fail-closed diagnostics.

Learning-status schema 2 reports each speed node's total/base/moving/
breakaway/authority populations, train/validation state, and—only when a fit
exists—the four observable candidate values above. It never labels the
retired rack gain or damping as learned. These values remain informational;
the UI cannot approve or activate them.

Both caches are cleared at manager start and published only after the offroad
owner validates its phase and current vehicle/build authority. They are
informational only: neither is evidence, approval, a profile,
controller-selection, or a safety input, and editing or deleting either one
cannot change steering. The `BLaTv2LifecycleStatus` schema and
`blatv2_profiled` implementation remain available for offline lifecycle
testing, but the stock-only field manager does not launch that process or
publish that cache.

### Pre-merge real-route audit

Routes b2 through b7 were replayed twice through the production importer with
the current vehicle-selected 409/4/7 envelope. Both passes produced identical
evidence (`8f67d9c41d9669f1a82f7abbe4f841c70434403a7235ad4236d3964fca40b81a`)
and manifest (`cc8e5387f8c83c6efc57b37e2bbe7d4d2e4651a985d1815d3483ebdd9c29b0d`)
hashes. The 10 and 15 m/s nodes qualified. The 5 m/s node had enough ordinary
evidence but correctly failed its independent full-authority validation; 0,
20, and 30 m/s remained evidence-limited or non-regressing. Every solvable
node selected the honest `static == kinetic` active-set boundary. Because the
six-node contract was incomplete, no candidate file was emitted and stock
selection was unchanged.

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
