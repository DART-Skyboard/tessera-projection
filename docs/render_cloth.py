"""Publication Cycles still of the Tessera cloth Cape-drape example.

    blender --factory-startup -b --python docs/render_cloth.py -- [outdir] [samples] [width] [height]

Env: CLOTH_SAMPLES (default 64), CLOTH_WIDTH (1920), CLOTH_HEIGHT (1080).
The still is written to <outdir>/cloth_cape_beauty.png (default: docs/renders).
"""
import os
import sys
import tempfile

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
# keep Cycles scratch off a tight workspace disk
os.environ.setdefault('TMPDIR', tempfile.gettempdir())

import arc_flow
arc_flow.register()
from arc_flow import cloth_ops as CO

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(HERE, 'renders')
SAMPLES = int(argv[1] if len(argv) > 1 else os.environ.get('CLOTH_SAMPLES', '64'))
W = int(argv[2] if len(argv) > 2 else os.environ.get('CLOTH_WIDTH', '1920'))
H = int(argv[3] if len(argv) > 3 else os.environ.get('CLOTH_HEIGHT', '1080'))

for nm in ('Cube', 'Light', 'Camera'):
    if nm in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)

sc = bpy.context.scene
print(CO.load_cape_example(sc), flush=True)
cl = sc.arc_flow_cloth
print("draping for beauty...", flush=True)
CO.drape_for_beauty(sc, gust_frames=50, settle_frames=40)
print("drape done, last %.1f ms" % cl.last_ms, flush=True)

os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'cloth_cape_beauty.png')
print("rendering", SAMPLES, "spp", W, "x", H, "->", path, flush=True)
CO.render_beauty(sc, path, samples=SAMPLES, resolution=(W, H), denoise=True)
print("wrote", path, os.path.getsize(path), "bytes", flush=True)
