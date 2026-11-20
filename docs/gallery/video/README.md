# Tessera Projection demo video

`make_video.py` records a ~60 s demo of the real Blender 5.3 UI (not a render): it builds and installs the extension zip into a clean
config, runs the Blender GUI under `xvfb-run` with software OpenGL, drives it from inside Blender (`bpy.app.timers`, `bpy.ops`),
grabs one real screen shot per state with ffmpeg `x11grab`, and assembles a 1920x1080 30 fps H.264 (yuv420p, crf 23, veryfast) mp4
with burned-in captions and title / free-software / end cards (PIL).

    python3 docs/gallery/video/make_video.py --blender /path/to/blender-5.3/blender --out tessera_demo.mp4
    # options: --work DIR  --target-seconds 60  --skip-build  --skip-capture (assemble existing frames)

Needs: Blender 5.3, ffmpeg with libx264, xvfb-run, python3 with Pillow, libX11/libXtst (X11 test extension, optional helpers).
The mp4 is not committed.

## Adding a scene
In `capture()` write a generator `sec_xxx()` that sets the scene up (demo operators, `view(...)`, `open_panels(<panel ids>)`) and calls
`yield from snap(title, subtitle, hold)` per state, then append `('xxx', sec_xxx)` to the `SECTIONS` storyboard list.
`VIDEO_SECTIONS=a,b` limits the run, `VIDEO_QUICK=1` shortens the loops, `VIDEO_FRAME_OFFSET=n` offsets frame numbers when merging runs.

## What is real and what is not
* Real: every frame between the title and the free-software card is a screen grab of the Blender window; the operators are the
  extension's own (Projection Remesh, Re-detect, Build, demo loaders, environment presets); the solver runs for real.
* Time-lapse: under software GL one grab takes seconds, so solver states are captured every 2-3 frames and each grab is held
  for a few output frames (about 5-10 screenshots per second of video).
* Scripted UI state: the N-panel expand state is not scriptable, so panel classes are re-registered with / without `DEFAULT_CLOSED`;
  the Tessera tab is selected via `region.active_panel_category`; camera/light icons are hidden with `overlay.show_extras`.
* The Fluids panel is not opened in Fluid mode (its draw() used an icon that does not exist in 5.3 in 0.4.0).
* Title, Free and open source and end cards are drawn with PIL, with the logo from `branding/tessera_1024.png`.

## Capture state (partial baseline, 0.5.0 RC)
Recorded and assembled: Projection Remesh (before, Symmetrize off, Symmetrize on), single-engine Systems, twin-engine crossfeed (enhanced and
rupture), lava lamp, a few colour-mixing frames, free-software and end cards (about 48 s).  Scenes written but not yet captured successfully:
colour mixing (full), crystal growth, steam, shader presets, Examples / Add Fluid Volume / viscosity + emission, wind tunnel with thermal
readback, Geometry Nodes sockets, Environment presets.  The capture is one Blender process per section with retries because a shared 8 GB
box OOM-killed the 2.5 GB GUI process; re-run `make_video.py` on an idle machine to produce the full video.
Known 0.5.0 RC issues seen while recording: thermal readback raises (shape mismatch in `rho_particles`) while the wind-tunnel emitter adds
particles (the scene sets `emit = False`); a red "StructRNA ... removed" line in the Tessera panel after scene changes.
