# c3x-chestnut

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

**Status: ⚠️ in progress; awaiting bench test and field validation**

## What it does

Makes the Chestnut big model usable on a comma 3X. Since upstream's `amd warp (#38684)` the
warp and the policy run as one graph on the Chestnut GPU, so every model run copies both
full road camera frames over the dock's USB link. A 3X camera frame is 1928x1208 NV12
(3,735,552 B each after skipping VisionIpc's trailing bytes), 7.47 MB per 20 Hz run, and on
a 3X that measured 64 ms median per inference against the 50 ms budget, so `modeldLagging`
kept openpilot from engaging. A comma four bins its cameras to 1344x760 and copes.

This branch warps the frames on the device's own GPU and sends only the warped model-size
frames (393,216 B) and the packed float inputs to the Chestnut: 395,384 B per run with the
small model's input shapes, about 0.46 MB if the big model carries spatial features.

## How it works

- Same routing upstream used before #38684: the warp runs on the local tinygrad backend
  (`QCOM` on the device) reading the VisionIpc buffers in place with `Tensor.from_blob`, and
  only the warped `2x6x128x256` uint8 tensor and the packed policy inputs move to `USB+AMD`.
- It stays one fused `run_model` JIT per camera resolution as in #38684; the JIT just spans two
  devices. The compiled pickle records the frame device in `input_devices['frame']`, and
  `modeld` reads it instead of assuming.
- `FRAME_DEV` selects this at compile time. Single-device builds leave it unset (`NPY`): frames
  are packed into the host input buffer exactly as upstream does today, so the small model on
  QCOM and the CPU/METAL builds are unchanged.
- The compile-time checks still mean something with two devices: random frames are written to
  host memory the frame device reads in place (the same path VisionIpc buffers take), and the
  capture replay and pickle round trip compare outputs and buffers across both devices.
- No toggles and no new params.

## Before driving

- tinygrad dedupes `BUFFER` UOps by `(slot, dtype, size, device)` when unpickling, and modeld
  keeps the big and small pickles loaded together, so a big-pickle `QCOM` buffer that matches a
  small-pickle `QCOM` buffer on all four would be silently shared. On a CPU stand-in (model on
  `CPU:1`, frames on `CPU`) the only `(dtype, size, device)` matches on the shared device are the
  read-only 9-float UV scale constants, which hold the same values in both pickles, and both
  models give bit-identical outputs loaded alone or together in either order. On the device,
  load each pickle in its own process, collect the `BUFFER` keys from every `run_model` JIT's
  `captured._linear`, and confirm no written `QCOM` buffer in one pickle matches the other.
- Check `modelV2.frameDropPerc` and the model execution time on a 3X with the Chestnut before
  engaging.

## What changed

- `openpilot/selfdrive/modeld/SConscript` — the Chestnut target compiles with
  `FRAME_DEV={tg_backend}` instead of the unused `FRAME_DEV=CPU`.
- `openpilot/selfdrive/modeld/compile_modeld.py` — `make_input_queues` takes a `frame_device`:
  `NPY` keeps the packed frames; any other device gets `frame` / `big_frame` JIT inputs wrapped
  with `from_blob`, and the packed buffer holds only the transforms and policy inputs. The warp
  runs on the frames' device, and the policy moves the warped frames to the model device. The
  pickle records `input_devices['frame']`.
- `openpilot/selfdrive/modeld/modeld.py` — with a frame device, `ModelState.run` caches one
  `from_blob` tensor per VisionIpc buffer and passes them as the frame inputs instead of copying
  the frames into the packed buffer.
