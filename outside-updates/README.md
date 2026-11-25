# Outside updates

This folder is not part of the official build or publish workflow.

`build.sh` only validates `arc_flow/` and writes `dist/tessera_projection-<version>.zip`.
`publish.sh` only uploads that `dist/` zip. Nothing in here is read by those scripts.

Each version subfolder is a self-contained drop you can download without touching source:

- the extension zip to upload on extensions.blender.org
- the still and the Blender UI shot from the review test

Current drop: [0.6.1](0.6.1/)

Earlier drops: [0.5.1](0.5.1/)
