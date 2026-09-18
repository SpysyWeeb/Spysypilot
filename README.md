# esp-active-debounce

Feature branch of [Spysypilot](https://github.com/SpysyWeeb/Spysypilot) — see the [`combo`](https://github.com/SpysyWeeb/Spysypilot/tree/combo) branch for the full fork overview. This fork is entirely vibe-coded, is a personal project, and is **not meant for others to use** — anyone is welcome to try it at their own risk.

## What it does

The "TAKE CONTROL IMMEDIATELY / Electronic Stability Control Active" alert no longer fires when the ESC flags a brief ABS intervention over a bump, a pothole or a pavement joint. It only fires once the car has reported ESC activity for 0.5 s without interruption, which a real skid or ABS stop does within its first half second.

## How it works

On the Palisade the ESC reports `TCS11.ABS_ACT`, which openpilot reads as `carState.espActive` and turns into the `espActive` soft-disable event. Two drives on 2026-09-16/17 (routes `0x82` and `0x87`) showed five such events, all ~0.36–0.39 s long, every one coincident with an IMU jolt of ~1 m/s² and none under hard braking: the ESC holds the flag for a fixed minimum on any wheel-speed blip. The soft-disable alert then shows for its 2 s minimum with the warning chime, and survives a driver cancel, for what was 0.4 s of ABS.

`CarEvents` now counts consecutive frames with `espActive` set and only adds the event once the count reaches 0.5 s (50 frames at 100 Hz). The count resets the frame the flag drops, so the event clears immediately when the ESC does. `carState.espActive` itself is unchanged and still logs the raw flag. A sustained intervention alerts 0.5 s later than before and the 3 s soft-disable countdown starts from that point.

## What changed

- `openpilot/selfdrive/car/car_events.py` — `espActive` is only added after the flag has been set for 0.5 s of consecutive frames.
- `openpilot/selfdrive/car/tests/test_car_events.py` — the recorded 0.36–0.39 s pulses never raise the event; a hold of 0.5 s does, and a release restarts the count.
