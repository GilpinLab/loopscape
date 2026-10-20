#!/bin/sh
set -e

git config core.hooksPath .githooks
git config remote.origin.push '+refs/public-snapshot/current:refs/heads/main'

tree=$(git rev-parse HEAD^{tree})
commit=$(git log -1 --pretty=%B | git commit-tree "$tree")
git update-ref refs/public-snapshot/current "$commit"

echo "Memoryless GitHub push configured."
