# Installation

This guide gives you a reproducible setup for local development and AI 100 execution.

## 1. Platform Prerequisites

Before installing this repository, complete the Cloud AI platform setup:

1. [Installation checklist](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/checklist.html)
2. [Platform SDK](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/platform-sdk.html)
3. [Prerequisites](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/prerequisites.html)
4. [Apps SDK](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/apps-sdk.html)

## 2. Clone the Repository

```bash
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
```

## 3. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 4. Install the Package

```bash
pip install -e .
```

If you are editing docs as well:

```bash
pip install -r docs/requirements.txt
```

## 5. Configure Artifact Cache Location (Optional)

By default, transformed artifacts are stored under `~/.cache/qeff_cache`.

You can override this:

- `QEFF_HOME`: preferred cache root for QEff artifacts.
- `XDG_CACHE_HOME`: fallback cache root when `QEFF_HOME` is not set.

Example:

```bash
export QEFF_HOME=/path/to/qeff_cache
```

## 6. Sanity Checks

Check package import:

```bash
python -c "import QEfficient; print('QEfficient import ok')"
```

Check CLI entrypoint is discoverable:

```bash
python -m QEfficient.cloud.infer --help
```

If your environment has dependency drift (for example `transformers` quantization API mismatch), align package versions to your target release branch before running compile/inference.

## Next Step

Continue with [Quickstart](quickstart.md) for the first end-to-end run.
