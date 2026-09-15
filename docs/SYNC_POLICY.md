# Sync policy

Rules for anyone or anything that merges upstream into this fork, rebuilds `combo`, or moves a submodule pin.
That includes the GitHub Actions in `.github/workflows/`, the nightly Hermes agent, and any coding agent.

## Why this exists

On 2026-09-14 `combo` did not build and would have crashed `hardwared` with a Chestnut attached. Every cause was a
process failure, not a code bug:

- An upstream merge on 2026-09-02 kept the fork's copy of five files the fork had never changed. That dropped
  upstream's same-day revert of a Chestnut change. From then on every upstream edit to those files conflicted and
  was resolved to "ours". By 2026-09-13 `hardwared.py` came from upstream and `chestnut/status.py` did not, and
  the two disagreed on argument order.
- Submodule pins were set by hand per branch. `combo` ended up with an opendbc where `safety_tick` takes no
  argument and a panda that passes one. Upstream openpilot still pinned the older opendbc. The fork had followed
  commaai/opendbc master past the pin set that upstream ships together.
- The rebuild pushed `combo` twice, seventeen minutes apart. Both pushes were red on the same error.
- 44 files were frozen at older upstream versions and 12 more matched no upstream commit at all. No fork commit
  had touched any of them.

The repair on 2026-09-14 restored those files from the merge base with commaai/openpilot master
(`5f50bda7b`). Restoring from the merge base gives the same result as taking upstream's tip only because `combo`
was one upstream commit behind, and only `openpilot/selfdrive/locationd/locationd.py` differed between the two.
On a branch that is further behind, restoring from the merge base keeps the branch internally consistent but still
stale. The fix for that is merging upstream, never copying files from upstream's tip.

## Rules

1. **Ownership allowlist.** `.github/upstream-parity-allowlist.txt` lists every path the fork intentionally
   changes, one line each, with the owning branch after ` # `. Lines are gitignore patterns anchored with a
   leading `/`. After any merge, a file that differs from the merge base with commaai/openpilot master and is not
   on the list is a bug. Fix it in the same commit, never carry it forward. If you add a fork-owned file, add its
   line in the same commit. `upstream-parity.yaml` enforces this on every pull request, every push to `combo`,
   and daily on `combo` plus the 20 rebuild branches. Feature branches that don't carry the list are checked
   against `combo`'s copy.
2. **Upstream wins where ours equals the base.** In a merge, a file the fork has not changed since the merge base
   always takes upstream's version. For a conflict in a file the fork owns, re-apply the fork's lines onto
   upstream's current version. Never keep an older upstream body.
3. **Reverts are honoured.** If the merged range contains a `Revert` commit, verify that the reverted content is
   absent from the result before pushing.
4. **Cross-file contracts are checked.** When a merge touches a caller or a callee (for example a process and
   the module it calls into), diff the signatures, import the modules, and run their tests.
5. **Submodule pins follow upstream openpilot's pins, as a pair.** `sync-submodules.yaml` is the only thing that
   moves the `panda` and `opendbc_repo` gitlinks on `combo` and `SOL`. It picks each target as follows
   (see `tools/sync/pin_target.sh`):
   - A submodule that follows commaai directly is pinned to exactly the commit that commaai/openpilot master pins.
   - A fork submodule (its `.gitmodules` entry names a tracked branch) is pinned to a commit found by walking the
     tracked branch's first-parent history, newest first. The target is the first commit whose merge base with
     commaai master is contained in the commit commaai/openpilot master pins. The walk stops at that first hit.
     It is not bounded by the current pin, because correcting a pin often means moving backward. It gives up
     after 1000 commits.

   The target is recomputed every run and set whenever it differs from the gitlink, forward or backward, so a
   hand-set pin is corrected. The panda `safety_tick` call and the opendbc declaration must agree. A changed pin
   is built and checked before it is committed, and it is pushed only if that passes. Otherwise, or when no
   consistent commit exists, the bot issue `[sync] submodule pins on <branch> need review` is opened and nothing
   is pushed. The fix for that issue is on the fork branch: merge commaai master only up to the pinned commit.
   Never set a gitlink by hand, and never advance to a fork-branch tip just because it is the tip.
6. **A red build is never pushed.** The rebuild runs the build and the engagement checks before pushing `combo`.
   If they fail, nothing is pushed, an issue is opened with the log, and no further rebuild runs until the cause
   is fixed. Read the previous run's CI result before starting.
7. **No hand edits inside merge commits.** A merge commit contains conflict resolution only. Adapting fork code to
   an upstream API change is a separate commit on the feature branch that names the upstream commit it adapts
   to.
8. **Every resolution is logged.** The body of a merge or rebuild commit lists the conflicted files, how each was
   resolved (ours, theirs or manual, and why), and the resulting pins.
9. **Feature code lives on its feature branch.** `combo` is only the merge of the feature branches plus the
   combo-level files: `README.md`, `.gitmodules`, `.github/`, `docs/`, `tools/sync/` and `AGENTS.md`. Code that
   exists only on `combo` is a bug: move it to the branch that owns it. The allowlist marks today's instances
   `combo-only`.
10. **Schema files are never resolved whole-file.** In `openpilot/cereal/log.capnp`, `openpilot/cereal/services.py`
    and `openpilot/common/params_keys.h`, take upstream's file and re-add each branch's entries. Ordinals stay
    unique and gap-free, and a removed ordinal is never reused.

## How to check a branch by hand

```
git fetch upstream master
tools/sync/parity_check.sh                      # checks HEAD against its merge base with upstream/master
tools/sync/parity_check.sh --rev origin/SOL \
  --fallback-allowlist .github/upstream-parity-allowlist.txt
```

Exit 0 means the branch differs from upstream only where the allowlist says it may. Exit 1 prints the offending
paths. For each one, restore it with `git show <merge-base>:<path> > <path>`, or add it to the allowlist with its
owning branch if the fork really owns it.

To see what pin the rule picks for a fork submodule (the submodule needs full history, and `commaai` must be a
remote inside it):

```
tools/sync/pin_target.sh opendbc_repo "$(git ls-tree upstream/master opendbc_repo | awk '{print $3}')" \
  refs/remotes/commaai/master refs/remotes/origin/<tracked-branch>
```
