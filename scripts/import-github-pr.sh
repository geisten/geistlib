#!/bin/sh
# import-github-pr.sh <number> — bring a pull request from the public GitHub
# mirror (github.com/geisten/geistlib) to git.geisten.net, where development
# happens: its head becomes the branch github-pr/<number> and a Gitea PR
# against main is opened.
#
# Read the PR on GitHub before running this. Pushing the branch starts the
# Gitea CI, which runs the contributor's code on our own machines.
#
# Merge the Gitea PR with a merge commit: once main is mirrored back, GitHub
# finds the PR's head on main and marks the GitHub PR as merged.
set -eu
n=${1:?usage: import-github-pr.sh <github-pr-number>}
gitea=${GITEA_REMOTE:-origin}

git fetch -q https://github.com/geisten/geistlib.git "+refs/pull/$n/head:refs/heads/github-pr/$n"
git push -q "$gitea" "github-pr/$n"
title=$(curl -fsS "https://api.github.com/repos/geisten/geistlib/pulls/$n" |
        python3 -c 'import json, sys; print(json.load(sys.stdin)["title"])')
tea pulls create --login git.geisten.net --repo geisten/geistlib \
    --head "github-pr/$n" --base main --title "$title (GitHub #$n)" \
    --description "Imported from https://github.com/geisten/geistlib/pull/$n by scripts/import-github-pr.sh. Merge with a merge commit, so GitHub marks #$n as merged once main is mirrored."
