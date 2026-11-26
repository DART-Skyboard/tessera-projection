# muscle log
- rebased onto 0.6.1 (e898366); MUSCLE tab added to ui_tabs/props enum
- muscle solver + props/ops/panel/gn + tests written; 5.3 and 5.2 pass; renders running
- stills: docs/renders/muscle_regions.png, muscle_face_regions.png, muscle_flex_before_after.png (Cycles), muscle_panel.png / muscle_panel_window.png (xvfb + opengl, Material Preview, ffmpeg x11grab; panel hosted in Item tab only for the screenshot because Python cannot switch the sidebar category)
- known: example skin is crude (head like a bottle, no hands/fingers); face regions on it are only indicative; modal paint/contour mouse handling not run headless; skin not wired to asset_colliders
