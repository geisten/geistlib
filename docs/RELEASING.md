# Releasing geistlib

The public release is a transaction, not a version string. A release is done
only when the exact `main` commit has passed the release gate, all platform
artifacts exist, the GitHub Release is published, and the postcondition check
passes.

## One-time repository settings

1. Create a protected GitHub Environment named `release` and require a
   maintainer approval for deployments to it.
2. Protect `main` with pull requests and required CI. Disable force pushes and
   do not grant an ordinary maintainer bypass for that rule.
3. Add a tag ruleset for `v*` that blocks updates and deletions without a
   bypass. Create release tags only through the release workflow. GitHub does
   not allow a personal repository to select the built-in GitHub Actions app
   as the sole creation bypass, so the workflow guard and postcondition enforce
   the creation policy while the ruleset makes published tags immutable.
4. Enable immutable releases before publishing. The workflow deliberately
   creates a draft, attaches and verifies every asset, and only then publishes
   it; publication makes the tag and assets immutable.

Repository settings are part of the release boundary. They cannot be enforced
by files in this checkout, so review them whenever release ownership changes.

## Prepare a release

Make one release-only pull request. It must:

- bump `GEIST_VERSION_*` and `GEIST_VERSION_STRING` in `include/geist.h`;
- set the same version in `CITATION.cff`;
- move every entry out of `[Unreleased]` into a dated version section;
- update the `[Unreleased]` and version comparison links;
- pass `make release-check` from a clean checkout.

Do not put a concrete release number in `README.md`. The README badge and links
resolve the latest published release from GitHub, so a source candidate cannot
masquerade as something users can download.

## Publish

After the release PR is merged and all required `main` checks are green:

1. Open **Actions → Release → Run workflow**.
2. Select `main`, enter the version without `v`, and leave `ref` as `main`.
3. Review and approve the protected `release` environment.

The workflow pins the current `main` SHA before any build. It then builds and
smoke-tests all platforms, creates or resumes a draft release for that exact
SHA, verifies the complete asset set, and only then publishes it as latest.
The tag is created by this workflow; never create or push a release tag by
hand.

Run `make release-state-check` to verify the public postconditions again. A
scheduled workflow runs the same check daily and after every successful release
workflow.

## Failure and retry

Failure before the publish job leaves no tag or release. Failure during the
publish job may leave a draft release and its tag; rerun the same version and
SHA to resume it. The guard refuses to overwrite a published release, a tag
pointing to another commit, or a non-`main` source.

Never delete or move a published tag to repair a release. Increment the patch
version and publish a new release instead.
