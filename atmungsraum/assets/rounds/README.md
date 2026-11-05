# Round-button illustrations

Generated with the built-in image generation tool on 2026-11-05, then traced
into genuine SVG paths. The original transparent PNGs are retained alongside
the runtime SVGs. Exact prompts and conversion settings are in `generation.json`.

- `probe-runde`: lungs with one repeat arrow for breathing practice.
- `haupt-runde`: a clipboard with three connected data points for research and
  data collection. It does not claim that the browser prototype collects data.

Both SVGs have fully transparent backgrounds, use the page's sage/cream palette
and share a 240×160 view box. The clipboard interior is transparent too.
Only the SVGs load on the page; there are no embedded bitmap images, external
references, scripts, fonts or animation. German button labels remain HTML text.
The illustrations are decorative supplements with empty alt text; the links'
accessible names come from their visible labels.

Conversion used the already installed VTracer 0.6.12 in Python 3.12. The local
Python 3.14 VTracer binding crashed even on a four-pixel input; no new package
or product dependency was installed. Palette/alpha classification removed
generation noise during tracing, without altering the original PNGs. Inspect
the resulting SVGs in the actual buttons when regenerating, including at
320 CSS px, enlarged text and short landscape sizes.
