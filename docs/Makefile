SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build

.PHONY: help html clean linkcheck serve

help:
	@echo "Targets: html, linkcheck, clean, serve"

html:
	DOC_VERSION=$${DOC_VERSION-main} \
	DOCS_VERSIONS_INDEX_URL=$${DOCS_VERSIONS_INDEX_URL-versions/index.html} \
	DOCS_LATEST_URL=$${DOCS_LATEST_URL-./} \
	DOCS_AVAILABLE_VERSIONS=$${DOCS_AVAILABLE_VERSIONS-main} \
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS)

linkcheck:
	DOC_VERSION=$${DOC_VERSION-main} \
	DOCS_VERSIONS_INDEX_URL=$${DOCS_VERSIONS_INDEX_URL-versions/index.html} \
	DOCS_LATEST_URL=$${DOCS_LATEST_URL-./} \
	DOCS_AVAILABLE_VERSIONS=$${DOCS_AVAILABLE_VERSIONS-main} \
	$(SPHINXBUILD) -M linkcheck $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

serve: html
	python -m http.server 8080 --directory $(BUILDDIR)/html
