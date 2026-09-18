# c3x-chestnut-v2

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress; rebuilt on upstream's split warp / precompiled model, awaiting a bench build and drives**

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
- Nothing is compiled on the dock any more: the big model is the precompiled LFS pickle upstream
  ships, and the `big_driving_warp_*` targets, their PCIe link wait in the SConscript and the
  `.chestnut.lock` are gone. `chestnut_compiled()` asks only for the model pickle and rejects an
  unfetched LFS pointer file, so the "Chestnut model not compiled" alert means the 776 MB file has
  not been fetched.
- Before the big model load starts, `modeld` polls `link_up()` once a second until the link is in
  L0, for up to `BIG_MODEL_TIMEOUT` (60 s) from the start of loading. The load thread only starts
  once the link is up and keeps its own 60 s timeout, so a load is never started after `modeld` has
  given up. If the link never comes up, no load starts: `ChestnutActive` goes False and the small
  model runs.
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
  7,471,616 B for the single-device layout. Not run on the device yet: a QCOM warp next to a USB+AMD
  model, where the readback of the warped frames and the order of the two uploads are what the
  bench must confirm.
- Bench-check the two-device path on the device offroad: load `ModelState(1928, 1208, True)` and run
  it on dummy frames, confirm `model_device` is the Chestnut and `warp_device` is `QCOM`, and that
  the upload to the model device is `npy_size + 393216` bytes.
- On the first drives with the dock: `modelV2.big` on every steady frame, `frameDropPerc` 0,
  `modelExecutionTime` well under 50 ms (the fused build gave 27 ms), no `modeldLagging` /
  `bigModelFailed`. The readback of the warped frames from the device GPU and the second upload are
  the only new costs against the fused build; if the execution time is above ~35 ms, look there.
- A cold start where `chestnutState.pcieLtssm` reaches `0x78` late: the first `chestnutGpuState`
  with `big` set follows it, no `bigModelFailed`. While the link is down, each power request can hold
  the dock's control endpoint for about 2 s, so `chestnutState` can have gaps while `modeld` waits.
- `op switch` to this code must fetch the ~776 MB `big_driving_tinygrad.pkl` through LFS. If the
  pointer file is all that arrives, `chestnut_compiled()` is false: the "Chestnut model not
  compiled" alert stays up offroad and `modeld` runs the small model without waiting for the link.

## What changed

- `openpilot/selfdrive/modeld/modeld.py` — `ModelState` loads the device's own warp pickle for
  every model, packs one upload per device, and moves the warped frames into the model upload when
  the devices differ. `main` waits for the Chestnut PCIe link with `link_up()` before starting the
  big model load and skips the load if the link is not up within `BIG_MODEL_TIMEOUT`.
- `openpilot/selfdrive/modeld/helpers.py` — `chestnut_compiled()` only checks for the big model
  pickle.
- `openpilot/selfdrive/modeld/SConscript` — no Chestnut targets: the warps are compiled for the
  device GPU only.
- `openpilot/system/hardware/chestnut/flash.py` — `link_up()` reads the LTSSM before the PCIe
  power request and skips the request when the link is already in L0.
