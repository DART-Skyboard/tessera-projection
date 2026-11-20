# Tessera Projection gallery

Every image is a real screenshot of Blender 5.3.0 running the extension zip, captured on a virtual X display (Xvfb, `--gpu-backend opengl`)
with synthesised mouse and keyboard input. Nothing is mocked up; the hero image is the Projection Remesh screenshot with a title block.

| File | Shows |
| --- | --- |
| `featured_1920x1080.png` | 1920x1080 hero image (Projection Remesh result with title block) |
| `01_projection_remesh.png` | Projection Remesh panel after the run: quad result with wireframe, Symmetrize options and the report |
| `01a_remesh_before_triangles.png` | The same mesh before remeshing (triangles) |
| `02_systems_single_engine.png` | Main panel and Systems list with the single-engine example running |
| `03_twin_crossfeed_enhanced.png` | Twin-engine example, crossfeed linked group, Systems status per engine |
| `03b_twin_crossfeed_failure.png` | Crossfeed "Bad" variant (hot chamber into the fuel tank) |
| `04a_environment_card.png` | Environment card |
| `04_environment_presets.png` | Environment planet presets drop-down open |
| `05_fluid_mode_lava_lamp.png` | Fluid mode panel with the lava lamp |
| `06_colour_mixing.png` | Colour mixing between fluids |
| `07_boolean_fluid_cavity.png` | Boolean projection, Build Fluid Cavity and its overlay |
| `08a_components_surfaces.png` | Components row: Container, Channel, Viscosity surface, Emission surface, Fluid volume |
| `08b_shader_presets.png` | Shader presets panel with the Glass preset applied |
| `08c_examples_dropdown.png` | Examples drop-down expanded, including Wind tunnel |
| `08d_wind_tunnel_thermal.png` | Wind tunnel with thermal readback in the viewport and the colour bar legend |
| `08e_transmutation_crystal.png` | Transmutation: crystal growth |
| `08f_transmutation_steam.png` | Transmutation: steam |
| `08g_geometry_nodes_wired.png` | Geometry Nodes editor: a Tessera group wired to stock nodes |
| `08h_geometry_nodes_add_tessera.png` | Geometry Nodes Add menu with the Tessera node group submenu |
| `09_cycles_render_window.png` | Render window: Cycles at low samples with a finished image |

## Re-running

```
GALLERY_ZIP=/path/to/tessera_projection-X.Y.Z.zip python3 docs/gallery/make_gallery.py all
python3 docs/gallery/make_gallery.py capture remesh twin   # only some scenes
python3 docs/gallery/make_gallery.py assemble              # copy raw captures here and build the hero image
```

Needs Blender 5.3 (path in the script), `xvfb-run`, ImageMagick `import`, python-xlib and Pillow. Raw captures go to `/tmp/tessera_gallery`.
To add a scene, write a `step_<name>` generator in the Blender section of `make_gallery.py` and add a line to `PUBLISH`.

## Notes

- Colours and particle shading in the viewport are viewport-only tweaks (object colour in Solid shading); they do not change the scene data.
- The twin-engine captures are taken at frame 22 for memory reasons on the shared machine, so the Systems rows still read RUNNING.
