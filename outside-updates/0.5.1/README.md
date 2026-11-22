# Tessera Projection 0.5.1 — platform review drop

Separate from the official `arc_flow/` workflow. Upload this zip in place of 0.5.0.

Review changes in this package only:

- tags are only Physics and Geometry Nodes
- the old `lotcbp` migration is removed (first version hosted on the platform)
- `assets_demo.py` is unchanged. It registers the three example assets under Fluid mode > Examples

## Download

- [tessera_projection-0.5.1.zip](tessera_projection-0.5.1.zip)
- [tessera_assets_render.png](tessera_assets_render.png)
- [tessera_ui.png](tessera_ui.png)

Direct raw links:

- https://github.com/DART-Skyboard/tessera-projection/raw/main/outside-updates/0.5.1/tessera_projection-0.5.1.zip
- https://github.com/DART-Skyboard/tessera-projection/raw/main/outside-updates/0.5.1/tessera_assets_render.png
- https://github.com/DART-Skyboard/tessera-projection/raw/main/outside-updates/0.5.1/tessera_ui.png

Frame 42 of the three-asset example, Blender 5.2.4. The test run used 2,200 particles so it would finish. The zip itself was not changed by that test.
- [tessera_wind_fighter.png](tessera_wind_fighter.png) — featured listing still: tailless fighter, open thermal wind, no tunnel cage
- [tessera_wind_fighter_ui.png](tessera_wind_fighter_ui.png) — widescreen Blender viewport of the fighter in thermal wind
- [tessera_wind_fighter.blend](tessera_wind_fighter.blend) — the fighter scene
- [tessera_wt_real.png](tessera_wt_real.png) — frame 8 of the Tessera wind-tunnel solver on the fighter, thermal skin from the add-on
- [tessera_wt_real.blend](tessera_wt_real.blend) — that wind-tunnel scene
- [tessera_combo_ui.png](tessera_combo_ui.png) — Cycles viewport, Tessera tab open, wind tunnel plus combustion on the fighter
- [tessera_combo.blend](tessera_combo.blend) — that scene (hidden RP-1, LOX and helium tanks, chamber, nozzle, solver streaks)
- [tessera_match_ui.png](tessera_match_ui.png) — Cycles viewport, Tessera tab, thermal readback on the fighter, solver wind
- [tessera_match.blend](tessera_match.blend) — that scene
