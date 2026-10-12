#!/bin/sh
# bench-publish.sh [TAG] — archive new bench-matrix runs and rebuild
# https://geisten.net/bench/ from the archive.
#
# The measure jobs upload each run to the web server. This collects the
# runs there into the geisten/bench-results repository (the archive; files
# are only ever added), optionally freezes the matrix for release TAG, and
# renders the site from the archive, so the server holds a view that can
# always be rebuilt. Exit 1 when tools/bench_matrix.py publish reports a
# regression; the archive and the site are updated either way.
#
# Needs BENCH_SSH_KEY (rrsync to the bench dir) and BENCH_RESULTS_KEY
# (deploy key with write access to geisten/bench-results).
set -eu
TAG=${1:-}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
(umask 077
 printf '%s\n' "$BENCH_SSH_KEY" > "$tmp/web_key"
 printf '%s\n' "$BENCH_RESULTS_KEY" > "$tmp/git_key")
web="ssh -F /dev/null -i $tmp/web_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$tmp/known_hosts"
export GIT_SSH_COMMAND="ssh -F /dev/null -i $tmp/git_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$tmp/known_hosts"

git clone -q _gitea@git.geisten.net:geisten/bench-results.git "$tmp/archive"
rsync -rlt --include='*/' --include='*.jsonl' --include='*.md' --exclude='*' \
    -e "$web" germar@www.geisten.net:runs "$tmp/archive/"
[ -z "$TAG" ] || python3 tools/bench_matrix.py snapshot "$tmp/archive" "$TAG"
git -C "$tmp/archive" add -A
if ! git -C "$tmp/archive" diff --cached --quiet; then
    git -C "$tmp/archive" -c user.name=bench-matrix -c user.email=bench-matrix@geisten.net \
        commit -qm "${TAG:+release $TAG: }runs up to $(date -u +%FT%TZ)"
    git -C "$tmp/archive" push -q origin HEAD
fi

mkdir "$tmp/site"
cp -R "$tmp/archive/runs" "$tmp/site/"
[ ! -d "$tmp/archive/releases" ] || cp -R "$tmp/archive/releases" "$tmp/site/"
rc=0
python3 tools/bench_matrix.py publish "$tmp/site" || rc=$?
rsync -rlt --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r -e "$web" "$tmp/site/" germar@www.geisten.net:
exit $rc
