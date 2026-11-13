#!/bin/bash
# leak_check.sh [RANGE]: scan the commits about to be pushed for personal details and credentials.
#
#   tools/leak_check.sh                 # the commits on HEAD that the fork's branch does not have yet
#   tools/leak_check.sh A..B            # a range
#
# It reads the added lines, the commit messages and the author and committer identities, and reports
#   - home-directory paths, email addresses other than GitHub's no-reply ones, and common credential formats
#   - anything matching the private patterns in the file $LEAK_PATTERNS names (one case-insensitive extended regex
#     a line, # for comments). That file holds the names and words that must not appear and is kept outside the
#     repository: the patterns themselves would be a leak here.
# Then gitleaks scans the same commits for secrets (its default rules), with the findings redacted.
# Exit 0 when clean, 1 on a hit, 2 when the check is incomplete: the private pattern file or gitleaks is missing.
range=$1
if [ -z "$range" ]; then
  branch=$(git rev-parse --abbrev-ref HEAD)
  up=$(git rev-parse -q --verify "fork/$branch" || git merge-base HEAD origin/main)
  range="$up..HEAD"
fi
generic='/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|sk-ant-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----'
email='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
email_ok='users\.noreply\.github\.com|@example\.(com|org)|git@github\.com'
bad=0
n=$(git rev-list --count "$range")
echo "leak check: $n commit(s) in $range"
ids=$(git log --format='%ae%n%ce' "$range" | sort -u | grep -v 'users\.noreply\.github\.com$')
if [ -n "$ids" ]; then echo "  commit identity is not a GitHub no-reply address:"; echo "$ids" | sed 's/^/    /'; bad=1; fi
added=$(git diff "$range" | grep -E '^\+' | grep -vE '^\+\+\+ ')
meta=$(git log --format='%an | %cn%n%B' "$range")
have_private=0
[ -n "$LEAK_PATTERNS" ] && [ -f "$LEAK_PATTERNS" ] && have_private=1
for src in added meta; do
  text=${!src}
  hits=$(printf '%s\n' "$text" | grep -n -i -E "$generic")
  [ $have_private = 1 ] && hits="$hits"$'\n'$(printf '%s\n' "$text" | grep -n -i -E -f <(grep -vE '^[[:space:]]*(#|$)' "$LEAK_PATTERNS"))
  hits="$hits"$'\n'$(printf '%s\n' "$text" | grep -n -i -E "$email" | grep -v -i -E "$email_ok")
  hits=$(printf '%s\n' "$hits" | grep -v '^$' | sort -u)
  if [ -n "$hits" ]; then
    [ $src = added ] && echo "  added lines match:" || echo "  commit messages or names match:"
    printf '%s\n' "$hits" | cut -c1-160 | head -20 | sed 's/^/    /'
    bad=1
  fi
done
have_gitleaks=0
if command -v gitleaks >/dev/null 2>&1; then
  have_gitleaks=1
  if ! out=$(gitleaks git --no-banner --no-color --redact -v --log-opts="$range" . 2>&1); then
    echo "  gitleaks reports secrets:"
    printf '%s\n' "$out" | grep -E '^(RuleID|File|Line|Commit):|leaks found' | cut -c1-160 | head -30 | sed 's/^/    /'
    bad=1
  fi
fi
if [ $bad = 1 ]; then echo "leak check: HITS. Do not push; fix the commits first."; exit 1; fi
if [ $have_gitleaks = 0 ]; then echo "leak check: patterns clean, but gitleaks is not installed: secrets were NOT scanned (brew install gitleaks)."; exit 2; fi
if [ $have_private = 0 ]; then echo "leak check: generic patterns clean, but LEAK_PATTERNS names no file: the private patterns were NOT checked."; exit 2; fi
echo "leak check: clean"
