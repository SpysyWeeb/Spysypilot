# BLaTv2 behavior replay memory contract

Status: **required before behavior qualification may run on comma hardware**.

Physical learning remains authoritative and stock remains active while this
work is incomplete.  A behavior-eligible cohort is authenticated with
file-backed summaries, but the device deliberately returns
`behavior_streaming_required` before constructing an eager
`RouteEvidenceArtifact`.  This is a memory-safety boundary, not a failed
Smooth/Swift/Strong verdict and not a request for more route data.

## Why eager replay is disabled

The canonical route-evidence files are compact.  Their current Python replay
representation is not:

1. `RouteEvidenceArtifact.from_file()` decodes every model publication,
   control witness, sparse publication, and event into Python objects.
2. The behavior decoder then creates a second whole-route object graph of
   physical frames, behavior intents, opaque controller inputs, and mappings.
3. The transaction retains all decoded routes, all reference samples, all
   segmentations, and the complete candidate job matrix in the parent.
4. Forked workers construct complete controller outputs and complete
   `BehaviorWindow` sample tuples, then return those frame-rich tuples to the
   parent for later reduction.
5. The second numerical authority repeats the load after the first authority
   has completed.

The measured CE evidence file was 146,363,115 bytes and occupied 909,200 KiB
after only step 1 (342,751 control witnesses and 68,528 model publications).
The authenticated summary inspector processed those same bytes at 20,824 KiB
peak RSS on the x86 audit host and returned the same artifact hash and counts.
That measurement validates descriptor selection; it does not make the eager
downstream decoder or transaction safe. The committed gate requires at least
four homogeneous routes. Retaining only the
first-stage artifacts would therefore consume roughly 3.5 GiB before the
larger decoded and replay graphs exist.  No compressed-file-size heuristic can
prove the downstream transaction safe, so there is intentionally no numeric
"small enough" production bypass.

## Required route-major design

The replacement must preserve the exact current numerical contract while
making memory independent of cohort duration:

1. **Descriptor-only selection.** Authenticate each immutable object with
   `RouteEvidenceFileSummary`.  The parent retains only route/source/hash/count
   metadata.  Cohort selection must never call `RouteEvidenceStore.load()`.
2. **Fresh authority reopen.** Authority 1 and authority 2 independently open
   the immutable path, verify the held file identity and section hashes, and
   create fresh decoder/controller state.  Neither authority may inherit a
   decoded object or cache from the other.
3. **One route and one view at a time.** A route child streams the fixed
   physical/control planes and incrementally decodes sparse model, delay,
   torque, maneuver, and event records.  It may hold only bounded look-ahead,
   the currently referenced sparse publications, and one controller state.
   It must not materialize full `ModelPublication`, `ControlsWitness`,
   `DecodedBehaviorRoute`, reference, or controller-output tuples.
4. **File-backed preparation.** Stateless references and neutral target
   samples are written to a private fixed-record scratch file while the route
   is authenticated.  Segmentation must consume that file incrementally or
   publish a bounded index of phase spans.  Any implementation which calls
   `tuple(samples)` over a complete route is still eager and does not satisfy
   this contract.
5. **Route-major variants.** For one route, replay exact stock, incumbent, and
   candidate policies in canonical identity order.  Reset the core before
   every variant.  Do not build a cohort-wide `_ReplayJob` tuple or expose the
   parent object graph through a fork-global registry.
6. **Bounded scoring result.** Score each selected phase window inside the
   route child.  Return only canonical `WindowMetricSet`/route-reduction
   records needed to reproduce `PolicyMetric`, never per-frame outputs or
   `BehaviorWindow.samples`.  The parent combines these bounded records in
   canonical route/variant order and retains only per-route metric values and
   coverage identities.
7. **Single local worker.** The comma backend runs exactly one route child at a
   time.  Workstation parallelism may be reconsidered only after child results
   are bounded and deterministic; it cannot change result order or bytes.
8. **Owned cleanup.** On offroad ownership loss, timeout, malformed output,
   artifact tamper, or child failure, terminate and reap the complete child
   process group, close held descriptors, unlink private scratch files, and
   publish no generation.  Authority 2 never begins after an incomplete
   authority 1.

The existing score and selection semantics are unchanged.  Whole-route
training/validation partitions, candidate ordering, exact-stock bootstrap,
segmentation hashes, coverage identities, paired-route uncertainty, and final
transaction JSON must remain byte-identical to the eager reference on bounded
fixtures.

### Bounded segmentation work authority

The file-backed route evaluator rejects, rather than truncates, a route which
exceeds any limit in the hashed segmentation configuration. Schema version 2
admits at most 65,536 raw phase spans, 4,096 qualifying phase windows, 4,096
event locators, and 65,536 event-to-phase attachments per route. These are
offline resource limits, not controller tuning values. They prevent the
one-million-frame wire limit from expanding into minutes of phase churn or an
unbounded Python descriptor graph; changing one creates a new segmentation
identity and therefore a new comparison population.

The committed gate can retain at most 240 route windows (six speed nodes ×
five maneuver classes × eight windows), before overlap between adjacent speed
nodes reduces that union. The 4,096-window discovery limit leaves more than
17× headroom over that useful output, the raw-span limit leaves another 16×
for short rejected phases, and the attachment limit allows 16 event links per
discovered window. Crossing any limit is evidence corruption/resource
exhaustion, never permission to choose a convenient subset.

Event lookup uses the ordered non-straight span index, and event attachment is
an interval-overlap pass. Work is proportional to spans, events, and actual
attachments, not their Cartesian product. Per-window metric scoring retains
only the same lowest identity-hash prefix that the existing per-route,
speed/class-stratum cap selects before reading metric values. This is a
streaming implementation of the existing value-independent selection rule,
not a new sampling policy.

## Acceptance required to remove the guard

- eager-versus-streaming A/A equality for every transaction byte on bounded
  fixtures and committed archived vectors;
- worker-count determinism on a workstation, with comma fixed at one worker;
- authority 2 proving a fresh reopen rather than reusing authority 1 state;
- parent RSS remaining within a fixed bound when route duration and cohort
  route count grow, aside from the explicitly bounded descriptor/result sets;
- a synthetic maximum-size route proving child RSS stays below the committed
  cap and child output stays bounded;
- tamper, truncation, abort, timeout, and child-crash tests proving cleanup and
  stock retention; and
- the complete behavior, evidence, generation, pipeline, and transaction test
  suites.

Until all of these pass, `behavior_streaming_required` is the correct and safe
terminal diagnostic for an otherwise eligible cohort.
