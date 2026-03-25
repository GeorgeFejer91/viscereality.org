# LaTeX Workspace

This workspace keeps editable LaTeX sources separate from rendered PDFs.

## Structure

- `templates/`: top-level `.tex` documents you edit directly
- `templates/shared/`: shared preamble and reusable layout pieces
- `build/`: intermediate LaTeX files (`.aux`, `.log`, etc.)
- `output/`: final PDFs with readable filenames matching the template name

## Naming

Use readable, document-specific filenames in `templates/`, for example:

- `praktikumsausschreibung-atemraeume.tex`
- `flyer-klinische-studie.tex`

The generated PDF will use the same base name in `output/`.

## Build

Render all templates:

```powershell
.\build.ps1
```

Render one specific template:

```powershell
.\build.ps1 -Template praktikumsausschreibung-atemraeume.tex
```
