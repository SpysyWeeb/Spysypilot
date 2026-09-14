#!/usr/bin/env bash
# Submodule pin target under the consistency rule (docs/SYNC_POLICY.md, rule 5).
#
# usage: tools/sync/pin_target.sh SUBMODULE_DIR UPSTREAM_PIN COMMAAI_MASTER_REF [FORK_BRANCH_REF]
#
#   UPSTREAM_PIN        the gitlink commaai/openpilot master carries for this submodule
#   COMMAAI_MASTER_REF  the submodule's commaai master, fetched inside SUBMODULE_DIR
#   FORK_BRANCH_REF     the tracked fork branch (from .gitmodules), fetched inside SUBMODULE_DIR;
#                       omit for a submodule that follows commaai directly
#
# Prints the target commit on stdout; diagnostics go to stderr.
#   commaai-direct: the target is UPSTREAM_PIN itself.
#   fork:           walk `git rev-list --first-parent FORK_BRANCH_REF` newest first and take the
#                   first commit C whose merge base(s) with COMMAAI_MASTER_REF are all ancestors
#                   of UPSTREAM_PIN, i.e. the newest fork commit that contains no commaai content
#                   newer than what upstream openpilot ships together.
#
# The walk stops at the first hit. It is not bounded by the current pin: the rule must be able
# to move a hand-set pin backward (on 2026-09-14 all three fork opendbc pins sat at their branch
# tips, 2-3 first-parent commits past the target). PIN_TARGET_MAX_WALK (default 1000) caps the
# walk; reaching the cap, or the root, is a failure rather than a guess.
#
# A shallow SUBMODULE_DIR can give a wrong merge base, so it is refused. PIN_TARGET_ALLOW_SHALLOW=1
# overrides that for a local dry run against an existing shallow checkout (a warning is printed);
# CI always unshallows first.
#
# exit: 0 target printed, 1 no consistent target, 2 usage or missing objects
set -euo pipefail

die() { echo "pin_target: $*" >&2; exit 2; }
[ $# -eq 3 ] || [ $# -eq 4 ] || die "usage: $0 SUBMODULE_DIR UPSTREAM_PIN COMMAAI_MASTER_REF [FORK_BRANCH_REF]"
max_walk=${PIN_TARGET_MAX_WALK:-1000}
dir=$1; pin_in=$2; commaai_ref=$3; fork_ref=${4-}
[ -n "$dir" ] && [ -n "$pin_in" ] && [ -n "$commaai_ref" ] || die "empty argument"
[ $# -eq 3 ] || [ -n "$fork_ref" ] || die "empty FORK_BRANCH_REF"
[[ "$max_walk" =~ ^[1-9][0-9]*$ ]] || die "PIN_TARGET_MAX_WALK must be a positive integer"

g() { git -C "$dir" "$@"; }
pin=$(g rev-parse --verify "$pin_in^{commit}") || die "upstream pin $pin_in is not present in $dir"
commaai=$(g rev-parse --verify "$commaai_ref^{commit}") || die "$commaai_ref is not present in $dir"

if [ -z "$fork_ref" ]; then
  echo "pin_target: $dir follows commaai directly; target = upstream pin $pin" >&2
  echo "$pin"
  exit 0
fi

g rev-parse --verify "$fork_ref^{commit}" >/dev/null || die "$fork_ref is not present in $dir"
if [ "$(g rev-parse --is-shallow-repository)" != false ]; then
  [ "${PIN_TARGET_ALLOW_SHALLOW:-0}" = 1 ] || die "$dir is a shallow clone; fetch full history first"
  echo "pin_target: WARNING: $dir is shallow; merge bases past the shallow boundary are unknown" >&2
fi

walked=0
while IFS= read -r c; do
  walked=$((walked + 1))
  if [ "$walked" -gt "$max_walk" ]; then
    echo "pin_target: no consistent commit in the newest $max_walk first-parent commits of $fork_ref" >&2
    exit 1
  fi
  bases=$(g merge-base --all "$c" "$commaai") || continue   # no common history with commaai: not a candidate
  ok=1
  for b in $bases; do
    g merge-base --is-ancestor "$b" "$pin" || { ok=0; break; }
  done
  if [ "$ok" = 1 ]; then
    echo "pin_target: $fork_ref first-parent #$walked $c; merge base with commaai master ${bases//$'\n'/ } is contained in upstream pin $pin" >&2
    echo "$c"
    exit 0
  fi
done < <(g rev-list --first-parent "$fork_ref")

echo "pin_target: no first-parent commit of $fork_ref is consistent with upstream pin $pin (walked $walked)" >&2
exit 1
