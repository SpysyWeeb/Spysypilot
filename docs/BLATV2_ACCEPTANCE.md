# BLaTv2 modular acceptance

This document defines when the modular controller may progress from an
offline artifact to shadow collection and, eventually, actuation. A later
stage may not waive an earlier one. Until every activation gate passes, the
stock openpilot torque controller remains the sole actuator.

## Artifact identity

Every report identifies:

- the openpilot commit;
- the opendbc commit;
- the panda commit;
- the controller-policy artifact;
- the learned vehicle-profile SHA-256;
- the learner-evidence SHA-256;
- the replay-harness commit and input-timeline hashes.

The controller and replay import the same numerical source. Device-specific
copies or replay-only implementations are not acceptable.

## Opendbc and panda boundary

The generic controller contains no vehicle torque-count constants. At
runtime it receives the detected vehicle's maximum torque, build rate,
release rate, command cadence, and driver-torque limits from opendbc.
Projection tests compare the controller-side predictor directly with
opendbc's production limiter arithmetic.

Opendbc remains the platform command owner, including platform-specific
request-bit fault avoidance. Panda remains the independent safety ceiling.
Neither layer may add a steering feel adjustment. A submodule pointer change
is part of the steering artifact and is never absorbed during a merge.

## Foundation gate

Before shadow collection:

1. time, intent, reference, rack mapping, plant, observer, learner, actuator,
   feasibility, bootstrap, and invalid-output contracts pass their unit
   suites;
2. the model-published scalar action timestamp survives a cereal round trip;
3. malformed or stale plans cannot expose a prior plan;
4. forward and inverse plant equations are reciprocal on deterministic
   synthetic traces;
5. feasible actuator requests are transparent, and infeasible projections
   exactly match opendbc;
6. an unqualified or mismatched vehicle profile cannot activate;
7. source scans find no BLaTv1 mechanism, platform limit literal, hidden
   timing reconstruction, or second torque-shaping path.

## Shadow gate

Shadow mode is structurally incapable of publishing `carControl`. In the
current field graph it is an offline replay/harness surface only; manager
does not launch `blatv2_shadowd` onroad.

Device and harness recomputation must be bit-identical for deterministic
fields. Environment measurements, such as compute time, are excluded from
bit-exact comparison and reported separately. The device timing authority is
the on-device measurement, not workstation replay.

An offline shadow run may update an in-process slow-learner preview. That
preview is non-durable and cannot change a live profile or controller
selection. Field evidence comes only from a closed full rlog that passes the
offroad importer below.

## Learned-profile qualification

Training and artifact writes occur only offroad. `blatv2_backfilld` is the
sole durable evidence writer. Evidence is bound to the vehicle identity,
speed-node grid, seed profile, profile schema, and learner schema.

The importer accepts only complete, closed full rlogs; qlogs are insufficient.
It may discover routes recorded before the importer existed, but each route
must pass exact reviewed build/schema provenance, dongle/vehicle identity,
CarParams, controller-envelope, sensor-resolution, segment-continuity, and
source-coverage checks. A route-local rejection cannot prevent a later valid
route from being considered.

Each eligible batch is replayed twice in fresh runtimes and both results must
be byte-identical. An authenticated ledger binds every accepted, rejected, or
late-skipped route to its content and disposition so a route is never counted
twice. Publication writes a complete immutable generation before atomically
switching its `CURRENT` pointer.

Each speed node independently requires its documented clean support,
excitation, chronological train/validation split, physically valid fit, and
held-out improvement. Samples affect only adjacent interpolation nodes.
Consequently, extended highway use cannot overwrite low-speed evidence.

A candidate profile is emitted only when every required node qualifies.
Partial profiles remain evidence, not control artifacts.

`BLaTv2LearningOperationStatus` may expose logger finalization, historical
scanning/replay progress, and terminal diagnostics. No managed onroad process
publishes live collection state. It is a clear-on-manager-start display
cache, never evidence, approval, or a controller-selection input.

The field manager must contain exactly one BLaTv2 process on a real car:
`blatv2_backfilld`, and its predicate must be offroad-only. The shadow,
live-learner, and profile-lifecycle adapters remain unregistered offline
tools while stock is the sole active controller. A future activation build
must first add a reviewed offroad witness for exact provisional-profile
exercise and feedback; it may not restore an always-on lifecycle observer by
assumption.

This isolation is a measured field-load decision. On route `d1` from combo
build `ff842`, `blatv2_shadowd` exited with status `-6` 302 times at roughly a
5.5-second cadence and averaged 26.18% of one CPU core; `blatv2_learnerd`
averaged another 20.55%. Reused-`Text` Cap'n Proto arena accumulation is one
possible explanation for the shadow failures, but it is a hypothesis rather
than a proven cause. The acceptance invariant follows from the observed load,
not from that hypothesis: manager launches zero BLaTv2 processes while
started/onroad.

## Replay promotion gate

A complete profile and explicit controller policy are evaluated with the
same canonical input timeline for every controller. Route work is
parallelizable, but aggregation order, result bytes, and hashes are fixed so
worker count cannot change a result.

At minimum, promotion reports:

- signed steady-hold path error;
- signed turn-in timing;
- release timing and overshoot;
- aggregate and worst-window torque-rate roughness;
- raw request, opendbc-projected, and delivered-path domains;
- named sharp-turn, unwind, handoff, and archived blip vectors;
- constraint duty, invalid-frame behavior, and observer saturation;
- stock baseline and the last accepted reference implementation.

No single aggregate may hide an uncompleted sharp turn, early corner entry,
late release, or high-speed oscillation. A metric that is undefined remains
explicitly undefined; its denominator and exclusions are reported.

The modular candidate must improve the intended failure class without
regressing another core value. “Smooth. Swift. Strong.” is a joint contract,
not a weighted score that permits trading one word away.

## Activation and rollback gate

An exact profile hash may be staged only when raw/applied replay,
delivered-curvature replay, deterministic A/A, comma-device timing, and
safety approval all identify that same hash and source pair. Each result is a
separate fail-closed field in the canonical approval artifact; omitted,
non-boolean, or false fields cannot activate. Staging and selection occur
offroad or at an engagement boundary; never mid-drive.

The first active drive is provisional. On the next offroad transition the
driver is asked whether steering felt **Better**, **About same**, **Worse**,
or **Not sure**. The answer is evidence:

- **Worse** requests rollback at the next engagement;
- **Better** or **About same** may clear provisional status only if all
  objective gates remain valid;
- **Not sure** keeps the profile provisional;
- steering overrides are bookmarks, not automatic negative labels.

Invalid core output follows the tested hold, decay, comm-issue, and
ten-valid-frame recovery contract. That safety behavior is not a tuning
surface.

## Branch and field status

Local development commits do not make a build field eligible. The feature
branch is pushed only after its current artifact clears the applicable
foundation, replay, safety, and worktree-identity checks. Combo remains
untouched until a separately reviewed merge preserves combo's submodule
pointers and proves that no out-of-scope controller changed.

The README remains **in progress** until the owner field-tests the accepted
artifact and explicitly authorizes a status change.
