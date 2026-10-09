# Verification

## Context and source

From `atmungsraum/`, run:

```powershell
./for-ai/scripts/check-context.ps1 -ProjectRoot .
```

The check verifies the subproject guidance, AI document set, empty HTML body,
black background declaration, and lack of external page dependencies or scripts.
After pushing the validated change to `main`, add `-RequireRemote` to verify that
the local repository HEAD matches the remote publishing branch.

## Rendered page

Inspect desktop and narrow mobile rendering, including 320 CSS px width. The
viewport should be entirely black with no visible text, elements, or scrollbars.
Confirm the body is empty and the computed root background is `rgb(0, 0, 0)`.
This page has no visible text, controls, animation, or font dependency; text-fitting,
focus-order, and reduced-motion cases have no runtime surface at this stage.

## Deployment

- Verify the GitHub Pages build reports `built` for the published commit, without
  a build error.
- Request both `/atmungsraum` and `/atmungsraum/`; a redirect from the first to the
  second is normal for a directory index.
- Confirm the final response is HTTP 200 and contains the expected page source,
  then inspect the public page in a browser.
- Verify the deployed `atmungsraum/for-ai/README.md` is reachable.

Report failed or pending checks explicitly; do not infer deployment from a push.

## Observed local verification — 2026-10-09

The context/source checker and Git whitespace check passed. Browser checks at
1280 × 720 and 320 × 568 CSS px showed an empty body with zero child elements,
`rgb(0, 0, 0)` as the root background, and no horizontal or vertical overflow.
Both rendered views were visually solid black. Deployment must still be checked
against the published commit using the procedure above.
