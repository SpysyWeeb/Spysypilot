# c3x-chestnut-v2

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress; drove well 2026-09-18; re-synced 2026-09-24 to upstream afa47035b (100 W GPU power limit, realtime scheduling after the load) with the loading alert and the standby warm-up added, awaiting a dock drive on that**

## What it does

Makes the Chestnut big model usable on a comma 3X. Upstream runs the model's crop warp on the
Chestnut GPU, so every 20 Hz model run copies both full road camera frames over the dock's USB
link. A 3X camera frame is 1928x1208 NV12 (3,735,552 B each after the trailing padding), 7.47 MB
per run, which measured 64 ms median per inference on this 3X (another 3X reported 51 ms and 7 %
dropped frames) against the 50 ms budget, so `modeldLagging` kept openpilot from engaging.
A comma four bins its cameras to 1344x760 and copes.

This branch warps the frames on the device's own GPU and sends only the warped model input
(2x6x128x256, 393,216 B) with the packed float inputs to the Chestnut: about 0.4 MB per run. The
first c3x-chestnut build did the same inside one fused warp+model compile and measured 27 ms
median on two drives; upstream has since split the warp from the model (#38864) and ships the
model precompiled (#38930), which is what this branch is rebuilt on.

It also keeps the first branch's fix for the start-up race: `modeld` does not give up on the big
model while the Chestnut's PCIe link is still training.

## How it works

- Upstream compiles the crop warp as its own tinygrad pickle per camera size. The one the small
  model uses (`driving_warp_1928x1208_tinygrad.pkl`, compiled for the device GPU) produces exactly
  the big model's `new_img` input, so `ModelState` loads that pickle for the big model too and reads
  the warp's device from it. When the warp and the model live on different devices, the camera
  frames and transforms are packed into an upload for the warp device, and the warped frames take
  the frames' place in the model device's upload: one host-to-device copy per device per run, the
  warp output moved between them through host memory. When both live on the same device (the small
  model, PC builds) the layout and the upload are byte for byte upstream's.
- Nothing is compiled on the dock: upstream ships the big model (#38930) and, since #38991, its two
  USB+AMD warps as precompiled LFS pickles, and compiles no Chestnut target itself any more. This
  branch loads the device GPU's `driving_warp_*` pickle for the big model too, so the shipped
  `big_driving_warp_*` files (about 0.9 MB each) are fetched but never loaded, and
  `chestnut_compiled()` asks only for the ~773 MB model pickle and rejects an unfetched LFS pointer
  file, so the "Chestnut model not compiled" alert means that file has not been fetched.
- Before the big model load starts, `modeld` polls `link_up()` once a second until the link is in
  L0, for up to `BIG_MODEL_TIMEOUT` (60 s) from the start of loading. The load thread only starts
  once the link is up and keeps its own 60 s timeout, so a load is never started after `modeld` has
  given up. If the link never comes up, no load starts: `ChestnutActive` goes False and the small
  model runs. The whole wait and load shows as a permanent "Big Model Loading" alert: the 3X onroad
  UI has no Chestnut indicator (the comma four's pulsing icon is mici-only) and upstream's event has
  only a no-entry alert, so on a 3X nothing was on screen unless the driver tried to engage. The
  alert is an idea from AmyJeanes' `tizi-to-mici` branch, written here from the fork's own facts.
- While the big model is active, the small model is warmed once at start-up (before the realtime
  priority is set), so a fallback's first run is not a 1.3 s graph build at realtime priority as it
  was on route 80.
- `link_up()` reads the LTSSM first and sends the PCIe power request (`0xF3`) only when the link is
  not in L0, the order tinygrad's `CustomASM24Controller` uses, so a link that trained by itself is
  not disturbed by the poll.
- No toggles and no new params.

## Before driving

- Done on a PC: a two-device stand-in, the small driving model compiled for `CPU:LLVM` (the
  SConscript's PC device) as the Chestnut model and the 1928x1208 warp compiled for tinygrad's
  `PYTHON` backend as the device GPU, both from the `driving_supercombo.onnx` and `compile_warp.py`
  command the SConscript uses. `ModelState` takes the split path, and over four runs of random
  frames, transforms and inputs its outputs are identical to the single-device path (warp and model
  both on `CPU:LLVM`); the warped bytes that reach the model device equal the warp's own output
  exactly. The model upload is 393,600 B (384 B of policy inputs plus the warped frames) against
  7,471,616 B for the single-device layout. The QCOM warp next to the USB+AMD model then ran on the
  2026-09-18 drive (combo d238dc098). Not yet driven: the 2026-09-24 sync (tinygrad 9d0446a4b with
  retargetable artifacts, the 100 W power limit, realtime scheduling after the load).
- Bench-check the two-device path on the device offroad: load `ModelState(1928, 1208, True)` and run
  it on dummy frames, confirm `model_device == 'AMD'` and `warp_device == 'QCOM'` (the artifacts
  record `Device.DEFAULT` at compile time; `load_oob` only chooses the device the big pickle is
  lowered for), and that the upload to the model device is `npy_size + 393216` bytes.
- Test the dock going away with the big model active (parked, bench onroad): cut the dock's 12 V,
  and separately pull its USB. tinygrad 9d0446a4b's hcq2 runtime waits on the USB bridge and on
  the AMD timeline without a timeout in places (`usb_drained`, `hcq_fence`), and the failed AMD
  target can stay in `Device['NPY'].pending`, so the review's code trace says a GPU loss may take
  30 s to raise, may never raise, or may make the small model's first upload wait on the dead
  device and exit `modeld`. Route 80's Idle Stop and Go restart (on the previous tinygrad) fell back
  in 4.5 s. Log the time to "big model failed" and whether the small model publishes afterwards. If
  it hangs, the fix belongs in tinygrad (bounded waits), not here.
- Rare paths to know: if the 60 s load join times out, the loader thread keeps using QCOM while the
  small model runs (fork-specific, since the big `ModelState` also owns the QCOM warp); and if
  `modeld` dies between setting `ChestnutLoading` and clearing it, the permanent alert stays for
  the drive (the param is cleared only on manager start and ignition transitions).
- The 100 W GPU power limit (upstream #38992) lets the SMU float the clocks instead of pinning them
  at max; a third party measured the big model 4.9 ms slower per run under it. On the next dock
  drive read `chestnutState.powerLimitW` (expect 100), and for `modelV2.big` frames keep the limit
  if `modelExecutionTime` p50 stays at or under 35 ms and p99 under 45 ms with `frameDropPerc` 0
  after warm-up and no `modeldLagging`; drop the one-line default in modeld.py if p99 goes above
  45 ms while `chestnutState.powerDrawW` sits near 95 W (the cap binding), or on any drops.
- On the first drives with the dock: `modelV2.big` on every steady frame, `frameDropPerc` 0,
  `modelExecutionTime` well under 50 ms (the fused build gave 27 ms), no `modeldLagging` /
  `bigModelFailed`. The readback of the warped frames from the device GPU and the second upload are
  the only new costs against the fused build; if the execution time is above ~35 ms, look there.
- A cold start where `chestnutState.pcieLtssm` reaches `0x78` late: the first `chestnutGpuState`
  with `big` set follows it, no `bigModelFailed`. While the link is down, each power request can hold
  the dock's control endpoint for about 2 s, so `chestnutState` can have gaps while `modeld` waits.
- `op switch` to this code must fetch the ~773 MB `big_driving_tinygrad.pkl` (plus the two unused
  ~0.9 MB big warps) through LFS. If the pointer file is all that arrives, `chestnut_compiled()` is
  false: the "Chestnut model not compiled" alert stays up offroad and `modeld` runs the small model
  without waiting for the link; the fix is `git lfs pull` in `/data/openpilot`, not the reboot the
  alert suggests.

## What changed

- `openpilot/selfdrive/modeld/modeld.py` — `ModelState` loads the device's own warp pickle for
  every model, packs one upload per device, and moves the warped frames into the model upload when
  the devices differ. `main` waits for the Chestnut PCIe link with `link_up()` before starting the
  big model load, skips the load if the link is not up within `BIG_MODEL_TIMEOUT`, and warms the
  standby small model while the big one is active.
- `openpilot/selfdrive/modeld/helpers.py` — `chestnut_compiled()` only checks for the big model
  pickle.
- `openpilot/selfdrive/selfdrived/events.py` — `bigModelLoading` gets a permanent "Big Model
  Loading" alert next to upstream's no-entry one.
- `openpilot/system/hardware/chestnut/flash.py` — `link_up()` reads the LTSSM before the PCIe
  power request and skips the request when the link is already in L0.
