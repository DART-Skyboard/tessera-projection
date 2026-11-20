"""Wind tunnel example with the emitter on: steps frames while particles are added, every per-particle array stays the same length and
thermal readback never raises.   blender --factory-startup -b --python tests/test_windtunnel_emit.py"""
import os, sys
import bpy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
arc_flow.register()
from arc_flow import windtunnel as WT, fluidmode_ops as FO
from arc_flow.core import fsurf
fails = [0]
def ok(c, m):
    print(("PASS " if c else "FAIL ") + m, flush=True)
    if not c: fails[0] += 1
print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
WT.load_example(sc)
FO.start(sc, play=False)
S = FO.RT.solver
n0 = S.n
bad = []
err = None
try:
    for f in list(range(1, 31)) + [8, 3, 12] + list(range(13, 41)) + [20, 41, 42, 43]:
        sc.frame_set(f)
        S = FO.RT.solver
        for name in fsurf.per_particle_names(S):
            pass
        mism = [k for k, v in vars(S).items() if isinstance(v, np.ndarray) and v.ndim >= 1 and k in ('x', 'v', 'T', 'fid', 'rho0_p', 'beta_p', 'cp_p', 'gasp') and v.shape[0] != S.n]
        if mism:
            bad.append((f, mism))
except Exception as e:
    import traceback; err = traceback.format_exc()
    print(err)
ok(err is None, "40 frames with emission and thermal readback raise nothing")
ok(not bad, "per-particle arrays stay in step with S.n %s" % bad[:3])
ok(FO.RT.solver.n > n0, "the emitter added particles (%d -> %d)" % (n0, FO.RT.solver.n))
FO.stop(sc)
print("FAILURES: %d" % fails[0])
sys.exit(1 if fails[0] else 0)
