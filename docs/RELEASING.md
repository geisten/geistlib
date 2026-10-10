# Releasing geistlib

The public release is a transaction, not a version string. A release is done
only when the exact `main` commit has passed the release gate, all platform
artifacts exist, they are downloadable and verified where users fetch them,
and the postcondition check passes.

Development and releases happen on `git.geisten.net` (`geisten/geistlib`,
private); `github.com/geisten/geistlib` is the public mirror and gets no new
releases (its old ones stay). Users download from:

| what | where |
| :-- | :-- |
| SDK archives, slim CLIs, `SHA256SUMS` | `https://geisten.net/download/geistlib/<tag>/`, `latest/` → the newest |
| BitNet-embedded CLIs (1.1 GB each) | `https://huggingface.co/geisten/geist-bitnet` — `main` is the newest, each release a tag; digests in `BITNET-SHA256SUMS` beside the others |

The Gitea release in the private repo holds the small files as the record.

## One-time settings

1. Organisation secrets on Gitea (`geisten`):
   - `DOWNLOAD_SSH_KEY` — may only rsync into
     `/var/www/htdocs/www.geisten.net/download` on the web server
     (`restrict,command="/usr/local/bin/rrsync -wo …"` in `authorized_keys`).
   - `HF_GEISTEN` — write access to `geisten/geist-bitnet` on Hugging Face.
2. Protect `main` on Gitea with pull requests and required CI; do not allow
   force pushes.
3. Release tags are created only by the release workflow; never push one by
   hand, never move or delete a published one.

## Prepare a release

Make one release-only pull request. It must:

- bump `GEIST_VERSION_*` and `GEIST_VERSION_STRING` in `include/geist.h`;
- set the same version in `CITATION.cff`;
- move every entry out of `[Unreleased]` into a dated version section;
- update the `[Unreleased]` and version comparison links;
- pass `make release-check` from a clean checkout.

Do not put a concrete release number in `README.md`: its links resolve
`latest/` and Hugging Face `main`.

## Rehearse, then publish

After the release PR is merged and all required `main` checks are green:

```sh
# rehearsal: builds everything, uploads to download/geistlib/rehearsal/ and the
# Hugging Face branch `rehearsal`, verifies, publishes nothing
tea actions workflows dispatch --login git.geisten.net --repo geisten/geistlib \
  --ref main -i version=0.21.0 -i ref=main -i dry_run=true release.yml
# the release
tea actions workflows dispatch --login git.geisten.net --repo geisten/geistlib \
  --ref main -i version=0.21.0 -i ref=main release.yml
```

The workflow pins the current `main` SHA before any build. The build jobs
(Pi: linux-arm64, desktop: linux-x86_64, Mac: macos-arm64) smoke-test every
artifact; the embedded CLIs go straight to the Hugging Face branch
`candidate-<tag>`. The release job then drafts the Gitea release, uploads the
version directory to geisten.net, verifies every file over HTTPS against
`SHA256SUMS` and the Hugging Face files against the built digests, and only
then publishes: the Gitea release (which creates the tag at that SHA), the
Hugging Face `main` commit and tag, and `latest/`.

Run `make release-state-check` to verify the public postconditions again; the
scheduled `release-state` workflow runs it daily.

## Failure and retry

Failure before the publish step leaves no tag and nothing under `latest/`:
fix and rerun the same version. A failure after the Gitea release is published
leaves the tag; finish by hand from the log (Hugging Face `main`, `latest/`),
never by moving the tag. To repair a published release, increment the patch
version and release again.
