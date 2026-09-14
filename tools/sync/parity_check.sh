#!/usr/bin/env bash
# Upstream parity check (docs/SYNC_POLICY.md, rule 1).
#
# Lists every path where <rev> differs from its merge base with commaai/openpilot master
# (the six submodule gitlinks excluded) and fails when a path is not covered by the
# allowlist. CI (.github/workflows/upstream-parity.yaml) and humans run this same script.
#
# usage: tools/sync/parity_check.sh [--rev REV] [--base SHA] [--allowlist FILE]
#                                   [--fallback-allowlist FILE] [--upstream-ref REF]
#                                   [--repo-slug OWNER/REPO]
#
#   --rev                 commit to check (default HEAD)
#   --base                merge base to diff against; skips the merge-base lookup
#   --allowlist           allowlist file (default: <rev>:.github/upstream-parity-allowlist.txt)
#   --fallback-allowlist  used when <rev> does not carry the allowlist (feature branches)
#   --upstream-ref        local ref of commaai/openpilot master (default upstream/master)
#   --repo-slug           this repo as OWNER/REPO for the GitHub compare API
#                         (default $GITHUB_REPOSITORY, else SpysyWeeb/Spysypilot)
#
# Merge base: --base if given; otherwise `git merge-base REV UPSTREAM_REF` when this clone has
# full history and the upstream ref; otherwise (shallow CI clones) the GitHub compare API
# (commaai/openpilot compare master...OWNER:REPO:SHA -> .merge_base_commit.sha, needs gh),
# after which the base commit is fetched with --depth 1 so its tree is available.
#
# Matching: allowlist lines are gitignore patterns; " # owner" comments are stripped and the
# patterns are handed to git's own matcher (git check-ignore --no-index) inside an empty
# throwaway repository, so this repo's .gitignore files and .git/info/exclude cannot leak in
# (in a real checkout, check-ignore also consults them, e.g. `*.pem` would "allow"
# openpilot/common/esim/gsma_ci_bundle.pem). Negated patterns (!) exclude a path again.
#
# exit: 0 clean, 1 violations, 2 usage or lookup error
set -euo pipefail

UPSTREAM_URL=https://github.com/commaai/openpilot.git
GITLINKS=(panda opendbc_repo tinygrad_repo msgq_repo rednose_repo teleoprtc_repo)
ALLOWLIST_PATH=.github/upstream-parity-allowlist.txt

rev=HEAD base="" allowlist="" fallback="" upstream_ref=upstream/master
slug="${GITHUB_REPOSITORY:-SpysyWeeb/Spysypilot}"

die() { echo "parity_check: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --rev) rev="${2:?}"; shift 2 ;;
    --base) base="${2:?}"; shift 2 ;;
    --allowlist) allowlist="${2:?}"; shift 2 ;;
    --fallback-allowlist) fallback="${2:?}"; shift 2 ;;
    --upstream-ref) upstream_ref="${2:?}"; shift 2 ;;
    --repo-slug) slug="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

git rev-parse --git-dir >/dev/null || die "not inside a git repository"
sha=$(git rev-parse --verify "$rev^{commit}") || die "cannot resolve --rev $rev"

if [ -z "$base" ]; then
  if [ "$(git rev-parse --is-shallow-repository)" = false ] && git rev-parse --verify -q "$upstream_ref^{commit}" >/dev/null; then
    base=$(git merge-base "$sha" "$upstream_ref") || die "no merge base between $sha and $upstream_ref"
    echo "merge base (git merge-base $upstream_ref): $base"
  else
    command -v gh >/dev/null || die "shallow clone or no $upstream_ref, and gh is not installed for the compare API"
    owner=${slug%%/*}; repo=${slug#*/}
    base=$(gh api "repos/commaai/openpilot/compare/master...${owner}:${repo}:${sha}" --jq '.merge_base_commit.sha') \
      || die "GitHub compare API failed for ${owner}:${repo}:${sha}"
    [[ "$base" =~ ^[0-9a-f]{40}$ ]] || die "compare API returned no merge base ('$base')"
    echo "merge base (GitHub compare API): $base"
  fi
fi
if ! git cat-file -e "$base^{tree}" 2>/dev/null; then
  git fetch --no-tags --depth 1 "$UPSTREAM_URL" "$base" || die "cannot fetch merge base $base"
fi
git cat-file -e "$base^{tree}" || die "merge base $base has no tree locally"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

if [ -n "$allowlist" ]; then
  [ -f "$allowlist" ] || die "allowlist $allowlist not found"
  cp "$allowlist" "$work/allowlist.txt"
elif git cat-file -e "$sha:$ALLOWLIST_PATH" 2>/dev/null; then
  git show "$sha:$ALLOWLIST_PATH" > "$work/allowlist.txt"
  allowlist="$sha:$ALLOWLIST_PATH"
elif [ -n "$fallback" ]; then
  [ -f "$fallback" ] || die "fallback allowlist $fallback not found"
  cp "$fallback" "$work/allowlist.txt"
  allowlist="$fallback (the checked commit carries no $ALLOWLIST_PATH)"
else
  die "$sha carries no $ALLOWLIST_PATH; pass --allowlist or --fallback-allowlist"
fi
echo "allowlist: $allowlist"

# drop " # owner" comments and comment/blank lines; what remains is plain gitignore syntax
sed -E -e 's/[[:space:]]+#.*$//' -e '/^[[:space:]]*(#|$)/d' "$work/allowlist.txt" > "$work/patterns"
[ -s "$work/patterns" ] || die "allowlist has no patterns"

# Full name-only diff, gitlinks dropped here rather than with ':(exclude)' pathspecs: a pathspec
# typo silently yields an empty list, which would pass. Same result as
#   git diff --name-only MB REV -- . ':(exclude)panda' ':(exclude)opendbc_repo' ...
git diff --no-renames --name-only -z "$base" "$sha" > "$work/all"
changed_count=0
: > "$work/changed"
while IFS= read -r -d '' path; do
  for g in "${GITLINKS[@]}"; do [ "$path" = "$g" ] && continue 2; done
  printf '%s\0' "$path" >> "$work/changed"
  changed_count=$((changed_count + 1))
done < "$work/all"

(
  unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR
  git init -q --template= "$work/matcher"
  set +e
  git -C "$work/matcher" -c core.excludesFile="$work/patterns" \
    check-ignore --no-index --stdin -z -v -n < "$work/changed" > "$work/matched"
  rc=$?
  set -e
  [ "$rc" -le 1 ] || { echo "parity_check: git check-ignore failed ($rc)" >&2; exit 2; }
)

total=0
violations=()
while IFS= read -r -d '' source && IFS= read -r -d '' _line && IFS= read -r -d '' pattern && IFS= read -r -d '' path; do
  total=$((total + 1))
  if [ -z "$source" ] || [ "${pattern:0:1}" = "!" ]; then
    violations+=("$path")
  fi
done < "$work/matched"

[ "$total" -eq "$changed_count" ] || die "matcher returned $total record(s) for $changed_count path(s)"
echo "checked $total path(s) that differ between $base and $sha"
if [ "${#violations[@]}" -gt 0 ]; then
  echo "::error::${#violations[@]} path(s) differ from upstream and are not in the allowlist" >&2
  printf '  %s\n' "${violations[@]}"
  echo "Restore each from the merge base (git show $base:<path> > <path>) or, if the fork owns it," \
       "add it to $ALLOWLIST_PATH with its owning branch. See docs/SYNC_POLICY.md."
  exit 1
fi
echo "parity OK: every differing path is allowlisted"
