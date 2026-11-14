# Atmungsraum Quest build assets

These files are the Experimenter inputs retained from the 0.1.29 Quest 2 build.
Use the [public build recipe](../README.md#build-from-the-public-repository).

| Path | Contents |
| --- | --- |
| `experimenter.app.json` | Experimenter feature selection, rendering and session settings. Particle paths are relative to this directory. |
| `experimenter.env.json` | Exact values for the selected features that read inputs from the build environment. |
| `provider/` | Hash-bound condition A and B audio, breathing guidance and session profile. |
| `particles/` | Viscereality compute shaders, particle positions, normals, dynamics data and the R8 mask texture used by the signed rehearsal APK. |
| `arm64-v8a/` | OpenXR loader and liblsl shared libraries from the signed rehearsal APK, with upstream notices. |

The retained authored scenario timelines and cues are tracked in
[`atmungsraum/assets/audio/`](../../assets/audio/). The assembly script stages
the canonical [`audio/box-loop.json`](../../audio/box-loop.json) and its four
phase cues as well as that library, then regenerates the scenario catalog.
One 705,600-frame cycle at 44,100 Hz gives four four-second phases. Its exact
JSON digest enters the feature-locked build environment as
`RUSTY_QUEST_NATIVE_RENDERER_BOX_LOOP_SHA256`; the selected Java source closure
includes both shared-loop audio classes. The app spec explicitly selects the
breathing setup manifest metadata. The feature lock binds the checkout's
actual source and asset inputs on every resolution.

In selected builds, preparation loops indefinitely, calibration observes two
complete cycles, and `haupt-15` starts a separate 900-second main run at a full
cycle boundary. Controller State is the fallback input on the same clock.
The old practice timelines and provider A/B assets remain intact for legacy
or unselected consumers; they do not replace the selected loop. Zero guidance
bias is supported; a nonzero Arm or live-bias request is explicitly unsupported.
The two [calibration menu prompts](../../assets/audio/calibration-menu/manifest.json)
are staged and checked independently of the scenario timelines. Their source
is Secret Tunnel v6 using the existing EmoDB Atmungsraum 3 voice profile
([Berlin EmoDB](https://github.com/audeering/emodb), Burkhardt et al., 2005).
Each MP3 has its exact approved German TXT alongside it.
These local menu voices remain available on unselected paths and cannot start
another player over the shared scheduler.

The [developer build flavor](../README.md#developer-and-rehearsal-signing)
changes the app identity and disables debugging. It requires an existing,
durable developer keystore outside Git and an explicit version code. Keep the
same key and increase the code for later uploads; preserve the rehearsal key.
The build assets contain no developer signing material. No new shared-scheduler
APK has been accepted yet; the 0.1.30 record remains historical provenance.

The provider schema calls these inputs a private asset provider because
Rusty Quest binds them through an app-specific asset inventory. The files
needed by this app are tracked here.

The ARM64 OpenXR loader matches the official Khronos
[1.1.63 Android release](https://github.com/KhronosGroup/OpenXR-SDK-Source/releases/tag/release-1.1.63)
and is distributed under the [Apache 2.0 license](arm64-v8a/OpenXR.LICENSE).
The ARM64 liblsl library was built for Android API 29 from
[`sccn/liblsl` 1.17.7](https://github.com/sccn/liblsl/tree/64988c6a14b8dc3b3f270ece58eab4f480bfab43).
See its [MIT license](arm64-v8a/liblsl.LICENSE) and
[staging notice](arm64-v8a/liblsl.NOTICE). The Rust LSL binding and other
Cargo dependencies are resolved by the pinned Rusty Quest source.

The [Rusty Quest license](https://github.com/MesmerPrism/rusty-quest/blob/99352620fd79ec5cdcee71b47c2ba4056241af16/LICENSE)
applies to the upstream native source and the corresponding Atmungsraum
patch. See the [APK record](../apks/README.md) for the full signed artifact,
source revision and verification evidence.
