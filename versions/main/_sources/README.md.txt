# Docs

This directory contains the instructions for building static html documentations based on [sphinx](https://www.sphinx-doc.org/en/master/).


## Build the docs
Install the packages required for building documentation:

```sh
 pip install -r docs/requirements.txt
```

And then, change directory to docs folder to build the docs.

```sh
cd docs/
# To build docs specific to branch
sphinx-build -M html . build
# [Optional] To build docs for all the supporting branches
sphinx-multiversion . build
```
## Preview the docs locally
 
```bash
cd build/html
python -m http.server
```
You can visit the page with your web browser with url `http://localhost:8080`.

## CI/CD automation

- PRs run `.github/workflows/docs-check.yml` to validate docs build.
- Pushes to `main` run `.github/workflows/docs-deploy.yml` and publish latest docs to `gh-pages` root.
- Pushes of tags matching `v*` publish a version snapshot at `versions/<tag>/` on `gh-pages`.
