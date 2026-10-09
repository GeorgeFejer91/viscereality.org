# Workflow

1. Read the root guidance and this control plane; inspect `git status` before edits.
2. Keep the local checkout sparse and the fetch blob-filtered. Select only this
   subproject and the necessary root guidance; do not clone the full media-heavy
   repository, run `git sparse-checkout disable`, or fetch large assets needlessly.
3. Base edits on the current remote `main`. Preserve unrelated changes and files
   outside the sparse checkout; absence locally does not mean deletion remotely.
4. Keep product code outside `for-ai/`. Update these notes and the parent
   `for-ai/subprojects.md` when the page contract changes.
5. Run `for-ai/scripts/check-context.ps1 -ProjectRoot .` from `atmungsraum/` and
   the rendered checks in [VERIFICATION.md](VERIFICATION.md).
6. Review and stage only intended paths. Publishing requires user authorization;
   the 2026-10-09 request authorizes publishing the initial blank page and its notes.
   Never force-push or bypass branch protections.
7. After publication, run the checker with `-RequireRemote`, check the GitHub Pages
   build for the pushed commit, and verify the public URL. A successful push alone
   is not evidence that the page is live.

Local setup shortcuts and `LOCAL-PROJECT.md` are excluded by `.git/info/exclude`.
Do not publish them or machine-specific paths.
