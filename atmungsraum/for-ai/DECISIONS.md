# Decisions

## 2026-10-09 — Initial page

The user requested `/atmungsraum` live on the existing GitHub Pages site, with a
subproject-local `for-ai/` folder, and specified a blank black page for now.
Use a static directory index with an empty body and a black document background.
Retain a descriptive browser-tab title without adding visible content.

## 2026-10-09 — Repository and workflow

Use the existing public `GeorgeFejer91/viscereality.org` repository and its verified
Pages source (`main`, `/`). Maintain a sparse, shallow, blob-filtered local checkout
because the user explicitly does not want the whole repository downloaded.

Adapt the `for-ai` skill's control-plane structure to this nested subproject.
Keep root guidance and the root subproject map as discovery routes. No separate
repository, site framework, build system, or future feature scaffold is needed.
