# for-ai

This folder is the repo-local memory for AI collaborators working on the Viscereality website and its related subprojects.

Read this folder first when starting work in this repository. Do not rely only on chat history. Start with this README, then read:

- [constraints.md](constraints.md) for active behavior, design, content, and workflow constraints.
- [subprojects.md](subprojects.md) for the current map of website subprojects and their expected behavior.
- [pptx-html-one-to-one-mapper.md](pptx-html-one-to-one-mapper.md) for the target workflow for object-based PowerPoint-to-HTML presentation conversion.
- [pptx-html-family-status.md](pptx-html-family-status.md) for the current three-deck shared-asset migration status.

## How To Use This Folder

- Treat these files as living constraints, not as historical notes.
- When a user gives a durable preference about the website or a subproject, update the relevant file in the same change.
- When a task changes the PPTX-to-HTML pipeline, presentation publishing flow, QA rules, asset policy, or known blocker list, update the relevant `for-ai` note before committing.
- Keep entries specific enough to test or review later.
- Prefer dated notes for decisions that may need revisiting.
- Do not duplicate large implementation details already clear from source code. Link to files or name the owning area instead.

## Scope

This folder should cover:

- Main website behavior and visual language.
- Presentation players and generated presentation assets.
- Research/publication content expectations.
- Media handling, performance, accessibility, and mobile behavior.
- Tooling or generation workflows that future AI agents are likely to touch.

It should not store generated media, build output, private credentials, or large project artifacts.
