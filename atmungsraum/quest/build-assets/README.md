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

The six authored scenario timelines and cues are already tracked in
[`atmungsraum/assets/audio/`](../../assets/audio/). The assembly script stages
them and regenerates the scenario catalog. It resolves the feature lock on
each developer's machine so the lock can bind that checkout's build inputs.
The two [calibration menu prompts](../../assets/audio/calibration-menu/manifest.json)
are staged and checked independently of the scenario timelines. Their source
is Secret Tunnel v6 using the existing EmoDB Atmungsraum 3 voice profile
([Berlin EmoDB](https://github.com/audeering/emodb), Burkhardt et al., 2005).
Each MP3 has its exact approved German TXT alongside it.

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
