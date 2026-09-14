1. Any agent (including the nightly Hermes rebuild): read [docs/SYNC_POLICY.md](docs/SYNC_POLICY.md) before any merge, rebuild, pin move or push.
2. Parity check: `tools/sync/parity_check.sh` must exit 0 before a push. A new fork-owned file gets its allowlist line (with owner) in the same commit.
3. A file the fork does not own takes upstream's version. Never keep an older upstream body.
4. Pins: `panda`/`opendbc_repo` follow commaai/openpilot master's pins through the consistency rule (`tools/sync/pin_target.sh`). Only `sync-submodules.yaml` moves them.
5. No hand-set pins. Never `git submodule update` in a linked worktree, and never commit a gitlink you picked yourself.
6. Schema: never resolve `log.capnp`, `services.py` or `params_keys.h` whole-file. Take upstream and re-add each branch's entries.
7. Schema: ordinals stay unique and gap-free. Never reuse a removed ordinal.
8. No hand edits inside merge commits. Adaptations are their own commits on the owning feature branch.
9. Never push red. Build and run the engagement checks first. If they fail, open an issue and push nothing.
10. Stage explicit paths (never `git commit -a`), and log every conflict resolution in the commit body.
