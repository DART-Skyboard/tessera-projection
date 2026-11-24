"""Headless Blender test of aerodynamic friction heating and blended airflow on the fighter:

    blender --factory-startup -b --python tests/test_windtunnel_friction_heat_bpy.py
"""
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import arc_flow
arc_flow.register()
from arc_flow import thermal as TM
from arc_flow import windtunnel as WT
from arc_flow.core import friction_heat as FH

fails = [0]


def ok(c, msg):
    print(("PASS " if c else "FAIL ") + msg, flush=True)
    if not c:
        fails[0] += 1


print("Blender", bpy.app.version_string, flush=True)
sc = bpy.context.scene
# a modest high-speed tunnel so the test stays quick but heating is measurable
WT.load_fighter_example(sc, wind=250.0, with_thermal=True)
air = bpy.data.objects.get("Fighter Airframe")
ok(air is not None, "fighter airframe exists")
ok(air.arc_flow_fm.collider and air.arc_flow_wt.body, "fighter is a tunnel collision body")
ok(air.arc_flow_th.on and air.arc_flow_th.aero, "aero friction heating is enabled")
ok('tessera_temp_k' in air.data.attributes and 'tessera_friction_heat' in air.data.attributes,
   "tessera_temp_k and tessera_friction_heat attributes written")
ok(air.data.color_attributes.get('af_thermal_color') is not None, "viewport colour attribute written")
ok(any((s.material and s.material.name.startswith("Tessera Aero Heat")) for s in air.material_slots)
   or any((s.material and s.material.name.startswith("Tessera Thermal")) for s in air.material_slots),
   "heat shader assigned")

T = np.empty(len(air.data.attributes['tessera_temp_k'].data), np.float32)
air.data.attributes['tessera_temp_k'].data.foreach_get('value', T)
Q = np.empty(len(air.data.attributes['tessera_friction_heat'].data), np.float32)
air.data.attributes['tessera_friction_heat'].data.foreach_get('value', Q)
ok(np.isfinite(T).all() and np.isfinite(Q).all() and not np.isnan(T).any() and not np.isnan(Q).any(),
   "vertex attributes are finite (no NaN/Inf)")
amb = float(sc.arc_flow.env.temperature)
ok(float(T.min()) >= amb - 0.5 and float(T.max()) > amb + 2.0,
   "temperature rises above ambient (%.2f .. %.2f K, ambient %.2f)" % (T.min(), T.max(), amb))
ok(float(Q.max()) > 100.0, "friction heat flux is written (max %.0f W/m2)" % Q.max())

# leading edges vs tail from the same surface samples the model wrote (world space)
st = TM.STATE.get(air.name)
ok(st is not None and st.P is not None and getattr(st, 'Ts', None) is not None, "thermal state stores sample temperatures")
if st is not None and st.P is not None:
    xs = st.P[:, 0]
    nose, tail = xs <= np.quantile(xs, 0.12), xs >= np.quantile(xs, 0.88)
    # available (cold-wall) heat flux from the free stream: peaks at leading edges, falls downstream
    # (the residual flux after a long wall soak is small everywhere as T -> T_aw)
    deps = bpy.context.evaluated_depsgraph_get()
    P, N, area, sel, size, L = TM.sample_geometry(air, deps, air.arc_flow_th.domain)
    ae = FH.aero_evaluate(P, N, v_inf=(float(sc.arc_flow_wt.wind), 0, 0), T_inf=amb, rho=1.225, mu=1.8e-5,
                          cp=1005.0, k=0.0257, L=L, T_wall=np.full(len(P), amb))
    ok(ae['q'][nose].mean() > ae['q'][tail].mean() * 1.2,
       "available heat flux peaks at leading edges and cools downstream (nose %.0f, tail %.0f W/m2)" %
       (ae['q'][nose].mean(), ae['q'][tail].mean()))
    ok(st.Ts[nose].mean() > st.Ts[tail].mean(),
       "nose stays hotter than the tail after heating (T %.2f vs %.2f K)" % (st.Ts[nose].mean(), st.Ts[tail].mean()))

# heating rises with wind speed
T0 = T.copy()
Q0 = Q.copy()
TM.apply_aero_heat(sc, air, v_inf=(400.0, 0.0, 0.0), steps=24, speedup=80.0)
T2 = np.empty_like(T)
air.data.attributes['tessera_temp_k'].data.foreach_get('value', T2)
Q2 = np.empty_like(Q)
air.data.attributes['tessera_friction_heat'].data.foreach_get('value', Q2)
ok(float(T2.max()) > float(T0.max()) + 1.0 and float(Q2.max()) > float(Q0.max()),
   "higher wind raises surface temperature and heat flux (T %.2f -> %.2f K, q %.0f -> %.0f W/m2)" %
   (T0.max(), T2.max(), Q0.max(), Q2.max()))

vol = bpy.data.objects.get(WT.AIRFLOW_NAME)
ok(vol is not None and len(vol.data.vertices) > 100, "blended airflow volume exists (%d points)" % (len(vol.data.vertices) if vol else 0))
if vol is not None and 'af_temperature' in vol.data.attributes:
    Tv = np.empty(len(vol.data.attributes['af_temperature'].data), np.float32)
    vol.data.attributes['af_temperature'].data.foreach_get('value', Tv)
    ok(np.isfinite(Tv).all() and float(Tv.max()) > amb,
       "slipstream carries a heated boundary layer (max %.1f K)" % Tv.max())

# existing VTOL thermal path still writes af_temperature when aero is on
# (a quick cube body: apply_aero_heat only)
print("ALL PASS" if not fails[0] else "FAILED %d" % fails[0])
sys.exit(1 if fails[0] else 0)
