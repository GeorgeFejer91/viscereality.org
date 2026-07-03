# Active Constraints

Last updated: 2026-07-02

These are the current durable constraints for `viscereality.org`. Update this file when the site's expected behavior changes.

## AI Workflow

- Preserve existing user changes. Check `git status --short` before editing and do not revert unrelated modified or untracked files.
- Keep changes scoped to the requested website or subproject behavior.
- Prefer the site's existing static HTML, CSS, and plain JavaScript patterns unless a subproject already uses another stack.
- For scientific or clinical claims, use cautious wording, keep citations attached to claims, and verify new evidence before adding it.
- When adding durable project rules, update this `for-ai` folder as part of the same change.

## Site Identity

- The site presents The Viscereality Project: bio-responsive VR, breathwork, interoception, exteroception, altered states of consciousness, and research collaborations.
- The tone should be research-grounded, immersive, and careful. Avoid unsupported medical or therapeutic promises.
- Use "Viscereality" consistently for the project brand unless quoting source material.

## Main Website

- The root page is a static site centered on `index.html`, `assets/css/site.css`, and `assets/js/site.js`.
- Keep the dark, media-rich, breath-responsive visual direction unless the user asks for a redesign.
- The desktop side menu and mobile bottom menu are important navigation affordances; test changes at narrow and wide viewport sizes.
- Favor progressive enhancement. Core content should remain readable even if heavy media, animation, or external embeds fail.
- Preserve lazy/deferred media behavior where possible. Large video and ASCII animation assets should not eagerly load unless they are first-viewport critical.
- Respect reduced-motion, coarse-pointer, and save-data behavior in `assets/js/site.js` when adding animation.

## Content And Evidence

- Scientific claims need nearby citations or clear framing as project hypotheses, aims, or exploratory design claims.
- Avoid overstating breathwork, VR, or altered-state effects as settled clinical outcomes.
- External links should open safely with `target="_blank"` and `rel="noopener noreferrer"` where appropriate.
- Image and video `alt` text or ARIA labels should describe the actual content or purpose, especially for research visuals.

## Media And Assets

- Keep media paths stable when published pages depend on them.
- Prefer optimized posters and lazy loading for video-heavy sections.
- Do not move or rename generated presentation chunks without updating their manifests and player references.
- Avoid committing private source paths, credentials, or machine-specific absolute paths into user-facing pages.

## Accessibility And Mobile

- Preserve keyboard-accessible controls for videos, presentation players, navigation, and interactive demos.
- Ensure touch targets remain usable on phones.
- Avoid text overlap, clipped labels, and layout shifts in the side menu, mobile menu, cards, and presentation controls.
- Check that interactive or animated elements have sensible paused, hidden, reduced-motion, or low-power states.

## Subproject Contract

- Subprojects may have their own local constraints. Add them to [subprojects.md](subprojects.md) when they become durable.
- Presentation pages should prioritize predictable playback, authored timing, and stable chapter navigation over decorative complexity.
- Tooling subprojects should keep generated output separate from source scripts and document the expected input/output folders.
- PowerPoint-to-HTML conversion should preserve PPT object identity and layering whenever possible. Avoid full-slide snapshots as the primary representation; use the smallest necessary raster fallback only for unsupported isolated objects/effects.
- In presentation Morph playback, objects inside a detected PowerPoint group or inferred panel/carousel frame must move with their parent group. The visual frame/border should remain above its children.
