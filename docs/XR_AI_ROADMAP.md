# XR AI roadmap (planned, not in 0.5.0)

Feasibility study for connecting Tessera Projection to NVIDIA's XR AI stack (upstream NVIDIA/xr-ai, public beta v0.4.0-beta, Apache-2.0; fork DART-Skyboard/xr-ai). Design only.

## 1. What the stack offers (verified from the fork's README, `docs/source` and package READMEs)

- **Hub (DeviceIOHub)** receives participant-tagged audio, video and data from web, Android, iOS/visionOS and native clients (LiveKit based, HTTPS and `wss://` on 8080) and fans events out to agent workers over **msgpack over ZMQ** (`xr-ai-hub-client`, deps: pyzmq, msgpack, Python 3.11-3.14). Frames are pulled on demand (`request_frame()`); a return path sends audio or data back to the participant.
- **Agent SDK** (Python): `xr_ai_tools` defines `Tool` / `AsyncTool` with Pydantic request and result models, run through bounded `run_tool_loop()` (an LLM selects tools); `xr_ai_voice` (STT, TTS), `xr_ai_models` (OpenAI-compatible clients), routing, runtime.
- **Samples**: simple-vlm, conversation, procedure-capture-guide, lab-instrument-monitoring, tea-making, xr-render-demo (voice-driven CloudXR scene edit over typed RPC).
- **CloudXR** optional remote rendering; models run locally (needs NVIDIA GPUs, hub host needs NVENC) or via NIM / external endpoints.

**Key finding:** there is no public HTTP, MCP or JSON tool endpoint. Tools are in-process Python objects inside an agent worker. The "typed capability services" are msgpack-over-ZMQ RPC, and the docs state this private transport "is not an agent API". The SDK packages are not on PyPI (editable repo-path sources; apps are built inside the xr-ai source tree). So an external program cannot simply register tools; a Python worker written against the SDK must do it.

**Unverified**: wire schema stability between betas; whether headset plane, mesh or depth data is exposed (the hub carries camera frames and data channels, so scene understanding would be our own data message or VLM-derived); CloudXR redistribution terms; any Blender or USD path.

## 2. Architecture options

| | A. In-Blender client | B. Local sidecar (recommended) |
|---|---|---|
| What | Extension speaks ZMQ/msgpack to the hub itself | Separate process, our own package `tessera-xr-bridge`, hosts the xr-ai worker and tools; talks to Blender over loopback |
| NVIDIA code | None, but we reimplement a beta private protocol | In the sidecar only, installed by the user from the fork; never in the extension |
| Blender deps | pyzmq/msgpack wheels bundled | Standard library only |
| Risk | Protocol drift breaks the extension | Drift breaks only the sidecar |

Choose **B**. The extension contains a small **bridge**: a loopback JSON-RPC server (127.0.0.1, random port, per-session token) that the sidecar calls, or a client of the sidecar; plus a modal timer that executes requests on the main thread. The sidecar defines one xr-ai `Tool` per Tessera capability, with schemas generated from our own `tessera_tools.json` manifest, and forwards calls to Blender. Requests are queued and run on the main thread by the timer.

Protocol: loopback HTTP+JSON with polling for job progress; binary payloads (meshes, textures) as PNG/glTF files in a cache directory, referenced by path plus hash.

## 3. Data flows

```
(a) voice -> tool
mic -> headset client -> Hub -> STT -> LLM tool loop -> sidecar Tool("tessera_remesh")
   -> loopback JSON {tool,args,token} -> Blender bridge timer -> bpy.ops.arcflow.* (main thread)
   -> result JSON {status, job_id, report} -> sidecar -> TTS -> Hub -> headset

(b) scene in, overlay out
headset camera frames + proportions (our data message / VLM estimate)
   -> Hub -> sidecar -> JSON {objects:[{name,bbox,pose,scale}]} -> bridge: create/update proxy objects
   -> Tessera solve (CPU) -> bake -> field/mesh/texture to cache dir (glTF or vertex-color PNG)
   -> sidecar -> Hub data return -> headset renders overlay on passthrough (headset-side shader)
```

Flow (b) needs a headset-side overlay renderer (our own client work); it is the largest unknown.

## 4. Performance reality check

The solver is CPU NumPy (about seconds per second of simulation at modest counts), nowhere near 72-90 Hz. The headset must never wait on Blender. Options, in order:

1. **Bake and stream**: simulate in short chunks, stream cached frames (temperature texture, particle positions); the headset plays back and interpolates.
2. **Low-res proxy**: coarse grid and fewer particles for interactive scenes, full resolution on demand, marked "preview".
3. **Field cache**: reuse the Fluid Cavity field when only the camera moves; re-solve on changed proportions.
4. **Async jobs**: tools return a job id at once; never block the UI thread beyond one step.
5. Later: optional GPU kernels if the proxy is too slow.

## 5. Blender extension platform rules

Verified from the manual's Getting Started page: the manifest may declare permissions `files`, `network`, `clipboard`, `camera`, `microphone`; each needs a reason of at most 64 characters, one short sentence, no final period; for network, code should also honour `bpy.app.online_access`. `license` is a required SPDX id (ours is GPL-compatible). Bundled third-party code goes in `wheels` as relative paths. The page says nothing about subprocesses or other bundled binaries. **Unverified**: the review policy on executables and opt-in network use (confirm before release).

Our rules: declare `network` and `files` (cache directory), reason e.g. "Loopback link to the optional XR assistant"; the bridge is **off by default**, starts only when the user enables it, binds loopback only; the sidecar starts only from an explicit "Start XR link" button running a user-configured command; no bundled binaries, NVIDIA code or models.

## 6. Security and privacy

Loopback bind plus per-session token; allow-list of tools (no `exec`, files only under the cache directory); scene-changing tools logged, one undo step each, optional confirmation for destructive ones; camera frames and voice not stored, cache deleted on stop; no telemetry. LLM output is untrusted: arguments validated against the manifest schema before any `bpy` call.

## 7. Licensing

xr-ai is Apache-2.0; models and CloudXR have their own terms (to review). The extension imports and ships none of it. The separate sidecar may depend on it with notices kept. No NVIDIA trademarks in button labels; name it only in docs, with attribution.

## 8. XR Guided Examples (Examples panel)

A "XR Guided Examples" list in the Examples panel. Entries are data files (steps: narration, Blender action, tool call, expected result). Two kinds: guides that launch NVIDIA's own samples through the sidecar (user installed; missing-dependency state shown) and our own recorded guides (tool-call logs plus captions, replayable without a headset). Buttons: "Run guide", "Next step", "Record guide".

## 9. Use cases and caveats

Chemist mixing beakers in AR (headset proportions, mixing and thermal overlay); hazmat spill (**labelled "visual illustration, not a safety prediction"** in the UI and every exported overlay); real aircraft overlay (anchor, scale and pose come from the client); voice control for all.

## 10. Phased plan and estimates

| Phase | Deliverable | Estimate |
|---|---|---|
| 0 | `tessera_tools.json` manifest and schema validator (pure Python, testable headless) | 2 days |
| 1 | **Minimal proof**: bridge (off by default, loopback, token, main-thread queue) exposing one operator (`remesh` with a voice phrase "remesh the selected object"), plus a ~150 line sidecar with one xr-ai voice tool; tested without a headset by a text query | 1-1.5 weeks |
| 2 | All tools (simulate, thermal readback, examples, systems), async jobs, progress, undo | 2 weeks |
| 3 | Scene ingest (proxy objects from client data) and overlay export (texture/glTF cache) | 3 weeks |
| 4 | Overlay client for passthrough (web-xr or visionOS), low-res proxy mode, caching | 4-6 weeks |
| 5 | XR Guided Examples, recorded guides, use-case packs, docs | 2 weeks |

Proof exit: a spoken or typed request runs one undoable Tessera operation and returns a spoken result; no socket exists while the bridge is disabled.

## 11. Sensors (camera, LiDAR, depth, other exposed hardware)

Goal: one button per sensor class in the bridge; sensors **off by default**, data stays on the machine unless the user connects the XR link.

**How each class is reached** (none of this is verified against the xr-ai docs unless stated; the hub itself carries camera frames and data channels, verified):
- Headset camera passthrough, depth, planes, meshes: OpenXR extensions (passthrough, scene understanding/plane/mesh, depth; availability is runtime and vendor dependent, unverified), Meta and other vendor SDKs, ARKit/RealityKit on iPhone/iPad/visionOS (LiDAR depth, plane and mesh anchors; visionOS restricts raw camera access to enterprise entitlements, unverified), ARCore Depth on Android.
- PC hardware: USB/UVC cameras (OpenCV/V4L2/AVFoundation), RealSense and Azure-Kinect-style depth cameras via their own SDKs, laser/LiDAR scanners over standard serial/Ethernet drivers, IMU, hand and eye tracking where the runtime exposes them.
- Blender Python has only the `camera` and `microphone` extension permissions and no LiDAR, depth or OpenXR sensor API. Therefore the **sidecar (or a headset client) reads the hardware** and streams results into Blender through the loopback bridge; the extension itself never opens sensor drivers.

**Formats**: point clouds and scene meshes as PLY/OBJ chunks or float32 numpy buffers (`.npy`) in the cache directory with a JSON header (units, frame, timestamp, pose); depth frames as 16-bit PNG or `.npy`; camera frames as JPEG/PNG. Large payloads by path and hash, small ones inline.

**Sensors panel**: device list with a Detect button (the sidecar reports what it found); per sensor a status line (idle, streaming, error, frames per second), an Enable toggle and a one-button start/stop (Camera, Depth/LiDAR, Hand/Eye, IMU, Other); an OS permission prompt is triggered by the sidecar/client, with a clear message when denied; a privacy notice naming what is captured, where it goes (this machine only, or the connected XR link) and a "Stop all" button; no recording unless the user presses Record.

**Mapping to Tessera**: scene meshes become collision bodies; plane detection sets the floor/ground and the Environment ground; depth gives object proportions and the enclosure for Environment/Systems detection; LiDAR point clouds are cleaned by Projection Remesh (decimate, then quad layout) before use; camera and depth frames also feed the XR AI vision tools through the sidecar.

**Limits**: consumer LiDAR is low-resolution (cm-level noise, short range, holes on glass and shiny liquids), meshes arrive in chunks at about 1-10 Hz, round trip over the bridge adds tens to hundreds of milliseconds, and remeshing a dense cloud runs in seconds on the CPU. Treat sensor geometry as approximate; scale and registration come from the device.

**Minimum viable step**: a Sensors panel with one Detect button and one UVC/USB camera toggle in the sidecar that saves a still into the cache and shows it as an image in Blender (about 3 days); next, import a single PLY point cloud from a file or depth camera and run Projection Remesh (about 1 week).
