"""Build the docs/render_friction_heat.py scenes (fighter wind tunnel + friction heat, cohesive water +
whitewater foam) and save each as a packed .blend instead of rendering, for the full-UI captures.

    blender --factory-startup -b --python docs/showcase/export_fh_blends.py -- <outdir>
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'render_friction_heat.py')
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else '/tmp/tessera_fh'
os.makedirs(OUT, exist_ok=True)


def _save_instead_of_render(**_kw):
    sc = bpy.context.scene
    path = os.path.join(OUT, os.path.splitext(os.path.basename(sc.render.filepath))[0] + '.blend')
    try:
        bpy.ops.file.pack_all()
    except Exception as ex:
        print('pack', ex)
    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True, compress=True)
    print('SAVED', path, flush=True)


code = open(SRC).read().replace('bpy.ops.render.render(write_still=True)', '_save_instead_of_render()')
sys.argv = [sys.argv[0], '--', 'all', OUT]
g = {'__file__': os.path.abspath(SRC), '__name__': '__main__', '_save_instead_of_render': _save_instead_of_render}
exec(compile(code, SRC, 'exec'), g)
