# Quest calibration attempt recording

Today the Quest consumes Polar acceleration to fit Personal Tilt and supplies
live ECG to the phone during preparation. The main writer opens at Arm and
only admits raw samples in the official main interval. The native main
manifest already identifies the retained input, classifier, calibration
generation and clock provenance. CSV v4 does not carry the precise input
method by itself.

## Implementation

- [x] Add app-private Quest storage for each accepted guided attempt. Use an
  attempt ID and the admitted command ID. Keep the UTC and monotonic recording
  anchors, original Polar and host time, connection and audio generations,
  planned and observed audio-frame boundaries, and loss and outcome evidence.
  Persist the raw ECG and XYZ acceleration before the calibration quality mask.
  Label the committed prefix of interrupted recordings.
- [x] Attach the writer to the native accepted calibration and typed Polar
  packet callbacks. Bound queueing and end-of-attempt delivery, keeping the
  sensor and renderer callbacks free of file I/O. Keep each repeated,
  superseded, failed or interrupted attempt retrievable. Preserve original
  timestamps and explicitly account for missing data.
- [x] Introduce an authenticated native preparation ID before the first
  attempt. Keep it through retries and the matching Arm. Retire it when the
  participant changes or preparation is abandoned. Preserve the native input
  accepted at Arm and the exact attempt that supplied its model. Official
  Start remains gated by the retained setup evidence. Old archives retain
  their existing representation.
- [ ] Define access and export alongside the existing verified data contract.
  Keep preparation records separate from the 900-second main CSV and its
  legacy recovery paths. In the meantime, keep new capture app-private
  and leave ambiguous preparation records unassigned.

## Acceptance

Exercise consecutive attempts, including a failed attempt and two successful
attempts for one participant. Select one for a main and reconcile the exact
attempt and input in the native evidence. Check explicit input changes to
pretrained Polar and Controller State. Verify a new participant cannot inherit
old attempts even when a participant code repeats. Simulate source replacement,
callback delay and process interruption, then compare original raw timestamps
and the recovered committed prefix. Finish with a paired-device test and
verify the separate main still records its original exact sample window.
