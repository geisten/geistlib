# Building & deploying

## Build locally

```sh
make                       # auto-detect target, MODE=release -> ./geist symlink
make TARGET=pi5 CC=gcc     # cross-target (see mk/target-*.mk)
make bin                   # all binaries (geist + eval/profile dev tools)
make test                  # unit + int + py (auto-fetches the model)
```

Binaries land in `bin/<target>/<mode>/tools/`. The user-facing CLI is a single
binary with subcommands:

| `geist` subcommand | what it is |
|---|---|
| `geist <model> <prompt>` | one-shot text completion (the release artifact) |
| `geist agent <model> <request>` | one-shot whitelist-gated tool loop |
| `geist chat <model>` | interactive chat + tools + memory palace (see [agent.md](agent.md)) |

The tool-use **agent** (`agent.h`) is a header-only library; the `agent` and
`chat` subcommands drive it in-process. The resident daemon is `--serve`:

```sh
geist -m model.gguf --serve /run/geist.sock        # external model
geist --serve /run/geist.sock                      # embedded model
```

The model stays warm on a **chmod-600 Unix socket**. Dynamic hosts send
newline-delimited JSON with a per-request toolset,
execute correlated `tool.call` frames themselves, return `tool.result`, and
receive `conversation.result`. Application adapters use this path, so the host
owns authorization and execution while the dynamic protocol consumes no HA
credentials. There is no REST/token fallback or line-protocol compatibility mode.

### Self-contained / dependency-free build

`GEMM_PROVIDER=native` drops the BLAS dependency; static linking drops libc.
This is exactly what `release.yml` ships:

```sh
# Linux: fully static against musl (no libc dependency at all)
make TARGET=linux CC=gcc GEMM_PROVIDER=native \
     EXTRA_CFLAGS=-D_GNU_SOURCE EXTRA_LDFLAGS="-static -s" \
     bin/linux/release/tools/geist
ldd bin/linux/release/tools/geist   # -> "not a dynamic executable"
```

### Single-binary (model baked in)

```sh
make EMBED_MODEL=gguf_artifacts/gemma4-e2b-Q4_K_M.gguf   # ./geist needs no model file
```
Only for small models — the binary grows by the model size and a >~1.5 GB binary
exceeds the 2 GB GitHub-release limit. For a 3 GB Gemma GGUF, keep the model **on
the server** and deploy only the binary (below). The GGUF must carry its own
tokenizer (gpt2-BPE / SPM / SPM-unigram — e.g. a BitNet TQ2_0 works).

`release.yml` ships an embedded single-file build **by default**: the
**`EMBED_MODEL_URL`** repo variable defaults to the BitNet 2B-4T GGUF, so every
release attaches a `geist-bitnet-<platform>` artifact (model baked in, no model
file) alongside the model-less `geist` CLI. Point it at a different small GGUF, or
unset it to ship the model-less CLI only.

## GitHub options

Only **release artifacts** is a wired-up workflow today; the container and
SSH-deploy rows are sketches (no such workflow exists yet).

| option | how | best for |
|---|---|---|
| **Release artifacts** ✅ `.github/workflows/release.yml` | push a `v*` tag → builds static `geist` + embedded `geist-bitnet` for linux-arm64/x86-64 (musl) + macos-arm64, attaches them to the GitHub Release | distributing the CLI; `curl`/`gh release download` on any box |
| **GHCR container** *(sketch — not wired up)* | an Actions job could build a Docker image and push to `ghcr.io/geisten/geist` for the server to `docker pull` | bundling binary + runtime; rollback by tag |
| **Actions → SSH deploy** *(sketch — see below)* | a workflow step `rsync`/`scp`s the binary to your server and `systemctl restart`s the service | deploying straight to your own server |

`release.yml` builds `tools/geist` — the one binary that carries the `agent` and
`chat` subcommands. Adding the future daemon is a one-line change to its build step.

## Deploying to geisten.net

The agent is **resident** (load the ~3 GB model once, keep it warm), so run it as
a long-lived service, not per-request. (This is the Gemma multimodal path; for a
text-only endpoint, `geist-bitnet` bakes the model into the binary — nothing to
ship to the server separately.) Recommended shape:

1. **Build** a musl-static binary in CI (as `release.yml` does) — no runtime deps.
2. **Ship the model once** to the server (e.g. `/srv/geist/model.gguf`); it is too
   large to bake in or attach to a release.
3. **Run as a systemd service** that holds the model warm. The shipped
   `--serve` daemon exposes a permission-gated Unix socket. A public HTTP/TLS
   front remains a separate interoperability layer.
4. **Deploy on tag** with an Actions SSH step:

```yaml
# sketch — add to a deploy workflow, needs SSH_KEY / HOST secrets
- run: |
    scp -i <key> bin/linux/release/tools/geist deploy@geisten.net:/srv/geist/geist
    ssh -i <key> deploy@geisten.net 'systemctl --user restart geist'
```

```ini
# /etc/systemd/system/geist.service  (model warm, restart on crash)
[Service]
# The resident agent daemon: model stays warm, one request per connection on a
# chmod-600 Unix socket (the daemon chmods it). -m gives the model path.
ExecStart=/srv/geist/geist agent -m /srv/geist/model.gguf --serve /run/geist/geist.sock
Restart=always
[Install]
WantedBy=multi-user.target
```

### The resident daemon

`geist agent -m model.gguf --serve /path/geist.sock` is implemented. It keeps
the model warm and accepts only host-neutral dynamic requests with correlated
call/result frames. `./geist serve ...` is not a
separate required process: the agent daemon is the serving surface.

> Security: prefer a Unix socket (`chmod 600`) or a localhost port behind nginx
> over a public listener. The agent's jailbreak resistance is the immutable
> offered set, schema/policy validation and global call budget
> ([agent.md](agent.md#security-model)), independent of transport.
