"""Realistic shader preset library (Tessera Projection): node-based materials built in code, for Cycles and EEVEE.

Every preset is a function that builds a material from named parameter nodes (``TS_<Name>`` Value / RGB nodes, drawn in the Shader
sub-panel and editable in the Shader Editor), optional attribute inputs and a Principled / Volume core.  Socket names are looked up at
runtime (Principled BSDF names changed in 4.x and 5.x: ``Transmission Weight`` vs ``Transmission`` ...), so the same code builds on
Blender 5.2 LTS and 5.3.

Fluid attributes (point attributes written by Fluid mode, readable on points as instancer attributes and on the surface mesh as geometry
attributes): ``fm_color`` tints transmitting / scattering materials (so mixed and transmuted fluids keep their colour), ``fm_temperature``
drives the lava, flame, smoke and metal glow (blackbody) and ``fm_density`` scales volumetric densities.  Each is blended with the preset's own
parameter by a ``Use fluid ...`` switch (0 on plain assets, 1 when the shader is assigned to a fluid), and falls back to the parameter where
the attribute is absent, so the same material also works on any mesh, curve or volume object.

Variants (``build(key, ..., preview=, particle=, grid=)``):
  * default: full quality (refraction + volume absorption inside closed meshes: the fluid surface / a container shaped mesh);
  * particle: for liquids drawn as points / instanced spheres: no refraction (a Fresnel specular over Transparent) with the absorption / scatter
    volume, so overlapping particle spheres accumulate their colours like one substance;
  * preview: cheap Principled-only look (alpha instead of refraction, no volumes) for viewport / EEVEE solid use;
  * grid: volume presets read the grid attributes ``density`` / ``temperature`` of a Volume object (OpenVDB) instead of a per particle falloff.
"""
import bpy

from .nodeutil import Builder, enable_nodes, inp, out

PREFIX = "TS "

# Principled BSDF socket names in older releases (the keys are the 5.x names)
_PRIN_ALIAS = {
    'Subsurface Weight': ('Subsurface',), 'Specular IOR Level': ('Specular',), 'Transmission Weight': ('Transmission',),
    'Coat Weight': ('Clearcoat',), 'Coat Roughness': ('Clearcoat Roughness',), 'Coat IOR': ('Clearcoat IOR',),
    'Sheen Weight': ('Sheen',), 'Emission Color': ('Emission',), 'Coat Tint': ('Clearcoat Tint',),
}

KINDS = {'SURFACE': "Surface", 'VOLUME': "Volume"}


class Ctx:
    """Build context: creates parameter nodes, attribute inputs and Principled nodes with runtime socket resolution."""

    def __init__(self, mat, over=None, preview=False, particle=False, grid=False, strict=False):
        self.mat = mat
        self.nt = enable_nodes(mat)
        self.nt.nodes.clear()
        self.b = Builder(self.nt)
        self.over = {k.lower().replace('_', ' '): v for k, v in (over or {}).items()}
        self.preview, self.particle, self.grid, self.strict = preview, particle, grid, strict
        self.unresolved = []
        self.params = {}
        self._y = 0

    # ----------------------------------------------------------------- parameters
    def _pos(self):
        self._y -= 140
        return self._y

    def p(self, name, default, kind=None):
        """Parameter node TS_<name>: float -> Value node, 3/4-tuple -> RGB node.  Returns its output socket."""
        v = self.over.get(name.lower(), default)
        color = isinstance(default, (tuple, list))
        if color:
            n = self.b.node('ShaderNodeRGB', label=name, x=-1400, y=self._pos())
            vv = tuple(v) if len(v) == 4 else (v[0], v[1], v[2], 1.0)
            n.outputs[0].default_value = vv
        else:
            n = self.b.node('ShaderNodeValue', label=name, x=-1400, y=self._pos())
            n.outputs[0].default_value = float(v)
        n.name = "TS_" + name
        self.params[name] = n
        return n.outputs[0]

    # ----------------------------------------------------------------- small node helpers
    def node(self, idname, label=None, **attrs):
        return self.b.node(idname, label=label, x=attrs.pop('x', -700 + 0), y=attrs.pop('y', self._pos()), **attrs)

    def set(self, sock, v):
        if v is None:
            return
        if hasattr(v, 'node'):
            self.b.link(v, sock)
        else:
            try:
                sock.default_value = v
            except (TypeError, ValueError):
                try:
                    sock.default_value = (v, v, v) if len(sock.default_value) == 3 else (v, v, v, 1.0)
                except Exception:
                    pass

    def math(self, op, a=None, b=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op)
        n.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is not None:
                self.set(n.inputs[i], v)
        return n.outputs[0]

    def mixc(self, a, b, fac):
        n = self.node('ShaderNodeMix', data_type='RGBA')
        self.set(inp(n, 'Factor'), fac)
        self.set(inp(n, 'A'), a)
        self.set(inp(n, 'B'), b)
        return out(n, 'Result')

    def mixf(self, a, b, fac):
        n = self.node('ShaderNodeMix', data_type='FLOAT')
        self.set(inp(n, 'Factor'), fac)
        self.set(inp(n, 'A'), a)
        self.set(inp(n, 'B'), b)
        return out(n, 'Result')

    def map_range(self, v, lo, hi, to_lo=0.0, to_hi=1.0, clamp=True):
        n = self.node('ShaderNodeMapRange')
        n.clamp = clamp
        self.set(n.inputs['Value'], v)
        self.set(n.inputs['From Min'], lo)
        self.set(n.inputs['From Max'], hi)
        self.set(n.inputs['To Min'], to_lo)
        self.set(n.inputs['To Max'], to_hi)
        return n.outputs['Result']

    def ramp(self, fac, stops):
        n = self.node('ShaderNodeValToRGB')
        els = n.color_ramp.elements
        while len(els) < len(stops):
            els.new(0.5)
        for e, (pos, col) in zip(els, stops):
            e.position = pos
            e.color = col if len(col) == 4 else (*col, 1.0)
        self.set(n.inputs[0], fac)
        return n.outputs['Color']

    def noise(self, scale, detail=6.0, roughness=0.55, coords='OBJECT'):
        tc = self.node('ShaderNodeTexCoord')
        n = self.node('ShaderNodeTexNoise')
        n.noise_dimensions = '3D'
        self.set(n.inputs['Scale'], scale)
        self.set(n.inputs['Detail'], detail)
        self.set(n.inputs['Roughness'], roughness)
        self.b.link(tc.outputs[coords.title() if coords != 'OBJECT' else 'Object'], n.inputs['Vector'])
        return n.outputs['Fac']

    def bump(self, strength, height, distance=0.02):
        n = self.node('ShaderNodeBump')
        self.set(n.inputs['Strength'], strength)
        self.set(n.inputs['Distance'], distance)
        self.set(n.inputs['Height'], height)
        return n.outputs['Normal']

    # ----------------------------------------------------------------- fluid attributes
    def _attr(self, name, color):
        """Sum of the instancer attribute (points / instances) and the geometry attribute (surface mesh): the one that is absent reads 0."""
        a1 = self.node('ShaderNodeAttribute', attribute_type='INSTANCER', attribute_name=name)
        a2 = self.node('ShaderNodeAttribute', attribute_type='GEOMETRY', attribute_name=name)
        if color:
            mx = self.node('ShaderNodeMix', data_type='RGBA', blend_type='ADD')
            self.set(inp(mx, 'Factor'), 1.0)
            self.b.link(a1.outputs['Color'], inp(mx, 'A'))
            self.b.link(a2.outputs['Color'], inp(mx, 'B'))
            return out(mx, 'Result')
        return self.math('ADD', a1.outputs['Fac'], a2.outputs['Fac'])

    def fluid_color(self, base, name="Use fluid colour"):
        """base colour, replaced by the fluid's per-particle colour (fm_color) where the attribute exists, scaled by the switch."""
        use = self.p(name, 0.0)
        ac = self._attr('fm_color', True)
        bw = self.node('ShaderNodeRGBToBW')
        self.b.link(ac, bw.inputs['Color'])
        present = self.math('GREATER_THAN', bw.outputs['Val'], 0.0005)
        return self.mixc(base, ac, self.math('MULTIPLY', use, present))

    def fluid_temp(self, default_k, name="Temperature", use_name="Use fluid temperature"):
        """Temperature in kelvin: the parameter, or fm_temperature where it exists."""
        t = self.p(name, default_k)
        use = self.p(use_name, 0.0)
        at = self._attr('fm_temperature', False)
        present = self.math('GREATER_THAN', at, 1.0)
        return self.mixf(t, at, self.math('MULTIPLY', use, present))

    def density_mult(self, ref=1000.0):
        """1, or fm_density / reference where the attribute exists and the switch is on (denser parcels are more opaque)."""
        use = self.p("Use fluid density", 0.0)
        r = self.p("Reference density", ref)
        ad = self._attr('fm_density', False)
        ratio = self.math('DIVIDE', ad, self.math('MAXIMUM', r, 1e-6))
        present = self.math('GREATER_THAN', ad, 1e-6)
        return self.mixf(1.0, self.math('MINIMUM', ratio, 20.0), self.math('MULTIPLY', use, present))

    def falloff(self):
        """Soft edge of an instanced unit sphere (object-space radius 1): 1 in the core, 0 at the rim.  Edge softness 0 = off."""
        soft = self.p("Edge softness", 0.0)
        if self.grid:
            return 1.0
        tc = self.node('ShaderNodeTexCoord')
        ln = self.node('ShaderNodeVectorMath', operation='LENGTH')
        self.b.link(tc.outputs['Object'], ln.inputs[0])
        d = self.math('SUBTRACT', 1.0, ln.outputs['Value'])
        f = self.math('DIVIDE', d, self.math('ADD', soft, 1e-4), clamp=True)
        fs = self.math('MULTIPLY', f, self.math('SUBTRACT', 3.0, self.math('MULTIPLY', f, 2.0)))   # f * (3 - 2f)
        fs = self.math('MULTIPLY', fs, f)                                                              # smoothstep-like f^2 (3 - 2f)
        off = self.math('LESS_THAN', soft, 1e-4)
        return self.math('MAXIMUM', fs, off)

    # ----------------------------------------------------------------- shader nodes
    def prin(self, vals=None, opt=None):
        """Principled BSDF with runtime socket-name resolution.  vals must resolve (recorded in unresolved), opt are set only if present."""
        n = self.b.node('ShaderNodeBsdfPrincipled', x=-200, y=0)
        for group, required in ((vals or {}, True), (opt or {}, False)):
            for k, v in group.items():
                s = self._find(n, k)
                if s is None:
                    if required:
                        self.unresolved.append(k)
                        if self.strict:
                            raise KeyError("Principled BSDF has no input %r (have %s)" % (k, [x.name for x in n.inputs]))
                    continue
                self.set(s, v)
        return n

    @staticmethod
    def _find(node, name):
        for nm in (name,) + _PRIN_ALIAS.get(name, ()):
            for s in node.inputs:
                if s.name == nm:
                    return s
        return None

    def volume_pr(self, **kv):
        n = self.b.node('ShaderNodeVolumePrincipled', x=-200, y=-400)
        for k, v in kv.items():
            k = k.replace('_', ' ')
            try:
                s = inp(n, k)
            except KeyError:
                self.unresolved.append(k)
                if self.strict:
                    raise
                continue
            if isinstance(s.default_value, str) if hasattr(s, 'default_value') else False:
                s.default_value = v
            else:
                self.set(s, v)
        return n

    def absorb(self, color, density):
        n = self.b.node('ShaderNodeVolumeAbsorption', x=-200, y=-700)
        self.set(n.inputs['Color'], color)
        self.set(n.inputs['Density'], density)
        return n.outputs[0]

    def scatter(self, color, density, aniso=0.0):
        n = self.b.node('ShaderNodeVolumeScatter', x=-200, y=-800)
        self.set(n.inputs['Color'], color)
        self.set(n.inputs['Density'], density)
        self.set(n.inputs['Anisotropy'], aniso)
        return n.outputs[0]

    def add_shader(self, a, b):
        n = self.b.node('ShaderNodeAddShader', x=100, y=-500)
        self.b.link(a, n.inputs[0])
        self.b.link(b, n.inputs[1])
        return n.outputs[0]

    def finish(self, surface=None, volume=None, displacement=None):
        o = self.b.node('ShaderNodeOutputMaterial', x=400, y=0)
        o.target = 'ALL'
        if surface is not None:
            self.b.link(surface, o.inputs['Surface'])
        if volume is not None:
            self.b.link(volume, o.inputs['Volume'])
        if displacement is not None:
            self.b.link(displacement, o.inputs['Displacement'])
        return o

    def particle_surface(self, tint, rough=0.0, ior=1.33, spec=1.0):
        """Particle variant surface: Transparent + a Fresnel glossy sheen (no refraction), so spheres add up through their volumes."""
        tr = self.b.node('ShaderNodeBsdfTransparent', x=-200, y=300)
        gl = self.b.node('ShaderNodeBsdfGlossy' if hasattr(bpy.types, 'ShaderNodeBsdfGlossy') else 'ShaderNodeBsdfAnisotropic', x=-200, y=200)
        self.set(gl.inputs['Roughness'], rough)
        self.set(gl.inputs['Color'], (1.0, 1.0, 1.0, 1.0))
        fr = self.node('ShaderNodeFresnel')
        self.set(fr.inputs['IOR'], ior)
        f = self.math('MULTIPLY', fr.outputs[0], spec)
        mx = self.b.node('ShaderNodeMixShader', x=100, y=300)
        self.b.link(f, mx.inputs[0])
        self.b.link(tr.outputs[0], mx.inputs[1])
        self.b.link(gl.outputs[0], mx.inputs[2])
        return mx.outputs[0]


# ============================================================================= builders
def _liquid(c, color, dens, ior, rough, scatter=0.02, aniso=0.0, coat=0.0, sss=0.0, ripple=0.0, ripple_scale=6.0, dye=0.18, spec_tint=None,
            dispersion=0.0, frost=0.0, thin=False):
    """Transparent liquid / glass: refractive Principled (transmission 1) over an absorption (+ scatter) volume inside the mesh."""
    col = c.fluid_color(c.p('Color', color))
    dn = c.p('Absorption density', dens)
    ior_s = c.p('IOR', ior)
    rg = c.p('Roughness', rough)
    nrm = None
    if ripple > 0:
        rs = c.p('Ripple strength', ripple)
        nz = c.noise(c.p('Ripple scale', ripple_scale), 5.0, 0.5)
        nrm = c.bump(rs, nz, 0.02)
    dm = c.density_mult()
    sc_ = c.p('Scatter', scatter)
    if c.preview:
        pr = c.prin({'Base Color': col, 'Roughness': rg, 'IOR': ior_s, 'Alpha': 0.55, 'Transmission Weight': 0.0,
                     'Specular IOR Level': 0.6, 'Coat Weight': coat})
        if nrm is not None:
            c.set(pr.inputs['Normal'], nrm)
        c.finish(surface=pr.outputs[0])
        return 'blend'
    if c.particle:
        surf = c.particle_surface(col, rg, 1.33 if ior < 1.4 else ior)
    else:
        base = c.mixc((1.0, 1.0, 1.0, 1.0), col, c.p('Surface tint', dye))
        opt = {'Transmission Dispersion Scale': dispersion} if dispersion else {}
        pr = c.prin({'Base Color': base, 'Roughness': rg, 'IOR': ior_s, 'Transmission Weight': 1.0, 'Specular IOR Level': 0.5,
                     'Coat Weight': coat, 'Coat Roughness': 0.0, 'Subsurface Weight': sss}, opt)
        if nrm is not None:
            c.set(pr.inputs['Normal'], nrm)
        if thin:
            pr.inputs['Thin Wall'].default_value = True
        if frost:
            fr = c.p('Frost', frost)
            nz = c.noise(40.0, 6.0, 0.6)
            c.set(pr.inputs['Roughness'], c.math('ADD', rg, c.math('MULTIPLY', nz, fr)))
        surf = pr.outputs[0]
    if thin:                      # a hollow vessel: the closed mesh is a shell, no medium inside
        c.finish(surface=surf)
        return 'refractive'
    d = c.math('MULTIPLY', dn, dm)
    vol = c.absorb(col, d)
    if scatter:
        vol = c.add_shader(vol, c.scatter(col, c.math('MULTIPLY', sc_, dm), aniso))
    c.finish(surface=surf, volume=vol)
    return 'refractive'


def b_water(c):
    return _liquid(c, (0.62, 0.82, 1.0), 1.6, 1.333, 0.0, scatter=0.01, ripple=0.0)


def b_sea_water(c):
    return _liquid(c, (0.1, 0.45, 0.5), 5.0, 1.34, 0.02, scatter=0.15, aniso=0.3, ripple=0.06, ripple_scale=3.0)


def b_glass(c):
    return _liquid(c, (0.9, 0.97, 0.95), 0.2, 1.5, 0.0, scatter=0.0, dye=0.1, dispersion=0.0)


def b_glass_thin(c):
    """Hollow glassware (lamp, beaker, bottle): a thin-walled glass shell, no refraction through the wall and no medium inside."""
    return _liquid(c, (0.92, 0.97, 0.96), 0.0, 1.5, 0.0, scatter=0.0, dye=0.05, thin=True)


def b_oil(c):
    return _liquid(c, (0.95, 0.72, 0.18), 5.0, 1.47, 0.04, scatter=0.02, coat=0.0, dye=0.25)


def b_honey(c):
    return _liquid(c, (0.95, 0.55, 0.08), 5.0, 1.49, 0.02, scatter=0.25, aniso=0.2, coat=0.2, sss=0.0, dye=0.4)


def b_syrup(c):
    return _liquid(c, (0.5, 0.2, 0.04), 5.5, 1.46, 0.015, scatter=0.12, aniso=0.1, coat=0.5, dye=0.5)


def b_ice(c):
    return _liquid(c, (0.72, 0.9, 1.0), 2.5, 1.31, 0.05, scatter=0.35, aniso=0.0, dye=0.15, frost=0.12, ripple=0.04, ripple_scale=9.0)


def b_crystal(c):
    """Gem / crystal: high IOR, faceted normals (per cell Voronoi), faint colour absorption, optional dispersion (5.3+)."""
    col = c.fluid_color(c.p('Color', (0.65, 0.35, 1.0)))
    dn = c.p('Absorption density', 3.0)
    ior = c.p('IOR', 1.75)
    rg = c.p('Roughness', 0.0)
    fct = c.p('Facet strength', 0.8)
    scale = c.p('Facet scale', 3.0)
    tc = c.node('ShaderNodeTexCoord')
    vo = c.node('ShaderNodeTexVoronoi')
    vo.feature = 'F1'
    c.set(vo.inputs['Scale'], scale)
    c.set(vo.inputs['Randomness'], 1.0)
    c.b.link(tc.outputs['Object'], vo.inputs['Vector'])
    sub = c.node('ShaderNodeVectorMath', operation='SUBTRACT')
    c.b.link(vo.outputs['Color'], sub.inputs[0])
    sub.inputs[1].default_value = (0.5, 0.5, 0.5)
    sc_ = c.node('ShaderNodeVectorMath', operation='SCALE')
    c.b.link(sub.outputs[0], sc_.inputs[0])
    c.b.link(fct, sc_.inputs['Scale'])
    ng = c.node('ShaderNodeNewGeometry')
    ad = c.node('ShaderNodeVectorMath', operation='ADD')
    c.b.link(ng.outputs['Normal'], ad.inputs[0])
    c.b.link(sc_.outputs[0], ad.inputs[1])
    nm = c.node('ShaderNodeVectorMath', operation='NORMALIZE')
    c.b.link(ad.outputs[0], nm.inputs[0])
    if c.preview:
        pr = c.prin({'Base Color': col, 'Roughness': rg, 'IOR': ior, 'Alpha': 0.7, 'Specular IOR Level': 1.0})
        c.set(pr.inputs['Normal'], nm.outputs[0])
        c.finish(surface=pr.outputs[0])
        return 'blend'
    if c.particle:
        surf = c.particle_surface(col, rg, 1.6, 1.0)
    else:
        pr = c.prin({'Base Color': c.mixc((1, 1, 1, 1), col, 0.2), 'Roughness': rg, 'IOR': ior, 'Transmission Weight': 1.0, 'Specular IOR Level': 0.7},
                    {'Transmission Dispersion Scale': 1.0})
        c.set(pr.inputs['Normal'], nm.outputs[0])
        surf = pr.outputs[0]
    c.finish(surface=surf, volume=c.absorb(col, c.math('MULTIPLY', dn, c.density_mult())))
    return 'refractive'


def b_milk(c):
    """Milk / cream: diffuse base with strong subsurface (radius larger in red), a soft coat; Cream raises the colour and roughness."""
    col = c.fluid_color(c.p('Color', (0.9, 0.88, 0.8)))
    rg = c.p('Roughness', 0.22)
    sss = c.p('Subsurface', 1.0)
    rad = c.p('Subsurface radius', (1.0, 0.65, 0.4))
    scale = c.p('Subsurface scale', 0.06)
    coat = c.p('Coat', 0.35)
    pr = c.prin({'Base Color': col, 'Roughness': rg, 'Subsurface Weight': sss, 'Subsurface Scale': scale, 'Specular IOR Level': 0.5,
                 'Coat Weight': coat, 'Coat Roughness': 0.05, 'IOR': 1.35}, {'Subsurface IOR': 1.35})
    c.set(pr.inputs['Subsurface Radius'], rad)
    c.finish(surface=pr.outputs[0])
    return 'sss'


def b_wax(c):
    col = c.fluid_color(c.p('Color', (1.0, 0.45, 0.2)))
    rg = c.p('Roughness', 0.32)
    sss = c.p('Subsurface', 0.85)
    scale = c.p('Subsurface scale', 0.03)
    # warmer, more translucent when hot: colour shifts toward a brighter tint with the temperature
    t = c.fluid_temp(300.0)
    warm = c.map_range(t, c.p('Cold temperature', 295.0), c.p('Hot temperature', 335.0))
    tint = c.mixc(col, c.mixc(col, (1.0, 0.85, 0.5, 1.0), 0.35), warm)
    rad = c.mixc((1.0, 0.45, 0.25, 1.0), (1.0, 0.8, 0.5, 1.0), warm)
    pr = c.prin({'Base Color': tint, 'Roughness': rg, 'Subsurface Weight': sss, 'Subsurface Scale': scale, 'Specular IOR Level': 0.6,
                 'Coat Weight': 0.18, 'Coat Roughness': 0.15, 'IOR': 1.45},
                {'Emission Color': col, 'Emission Strength': c.math('MULTIPLY', c.p('Warm glow', 0.15), warm)})
    c.set(pr.inputs['Subsurface Radius'], rad)
    c.finish(surface=pr.outputs[0])
    return 'sss'


def b_slime(c):
    col = c.fluid_color(c.p('Color', (0.28, 0.85, 0.08)))
    rg = c.p('Roughness', 0.12)
    sss = c.p('Subsurface', 0.7)
    gob = c.p('Goo bump', 0.25)
    nz = c.noise(c.p('Goo scale', 18.0), 6.0, 0.55)
    nrm = c.bump(gob, nz, 0.01)
    pr = c.prin({'Base Color': col, 'Roughness': rg, 'Subsurface Weight': sss, 'Subsurface Scale': 0.04, 'Specular IOR Level': 0.7,
                 'Coat Weight': 1.0, 'Coat Roughness': 0.03, 'IOR': 1.4, 'Transmission Weight': c.p('Translucency', 0.25)})
    c.set(pr.inputs['Normal'], nrm)
    c.set(pr.inputs['Subsurface Radius'], (0.6, 1.0, 0.3))
    vol = None if (c.preview or c.particle) else c.absorb(col, c.p('Absorption density', 6.0))
    c.finish(surface=pr.outputs[0], volume=vol)
    return 'sss'


def b_blood(c):
    col = c.fluid_color(c.p('Color', (0.42, 0.0, 0.01)))
    rg = c.p('Roughness', 0.06)
    sss = c.p('Subsurface', 0.55)
    pr = c.prin({'Base Color': col, 'Roughness': rg, 'Subsurface Weight': sss, 'Subsurface Scale': 0.02, 'Specular IOR Level': 0.7,
                 'Coat Weight': 0.8, 'Coat Roughness': 0.02, 'IOR': 1.38, 'Transmission Weight': c.p('Translucency', 0.3)})
    c.set(pr.inputs['Subsurface Radius'], (0.7, 0.04, 0.02))
    vol = None if (c.preview or c.particle) else c.absorb(c.mixc((0.8, 0.0, 0.0, 1.0), col, 0.5), c.p('Absorption density', 30.0))
    c.finish(surface=pr.outputs[0], volume=vol)
    return 'sss'


def _metal(c, color, rough, glow_default):
    col = c.fluid_color(c.p('Color', color))
    rg = c.p('Roughness', rough)
    t = c.fluid_temp(300.0)
    h = c.map_range(t, c.p('Glow starts at', 900.0), c.p('Glow full at', 1800.0))
    bb = c.node('ShaderNodeBlackbody')
    c.b.link(c.math('MAXIMUM', t, 800.0), bb.inputs['Temperature'])
    em = c.math('MULTIPLY', c.p('Emission strength', glow_default), c.math('MULTIPLY', h, h))
    pr = c.prin({'Base Color': col, 'Metallic': 1.0, 'Roughness': rg, 'Specular IOR Level': 0.5, 'IOR': 2.5, 'Emission Strength': em,
                 'Emission Color': bb.outputs[0]})
    c.finish(surface=pr.outputs[0])
    return 'metal'


def b_mercury(c):
    return _metal(c, (0.86, 0.88, 0.9), 0.02, 4.0)


def b_metal(c):
    return _metal(c, (0.95, 0.64, 0.45), 0.18, 8.0)


def b_lava(c):
    """Magma: emission from the temperature (blackbody), cooled crust broken up by object-space noise, rough dark crust where it is cold."""
    t = c.fluid_temp(1500.0)
    cold, hot = c.p('Cold temperature', 700.0), c.p('Hot temperature', 1500.0)
    heat = c.map_range(t, cold, hot)
    nz = c.noise(c.p('Crust scale', 5.0), 9.0, 0.62)
    crust = c.math('MULTIPLY', c.map_range(nz, 0.38, 0.62), c.p('Crust amount', 0.95))
    heff = c.map_range(c.math('SUBTRACT', c.math('MULTIPLY', heat, 1.15), crust), 0.05, 1.0)
    bb = c.node('ShaderNodeBlackbody')
    c.b.link(c.mixf(1000.0, c.p('Glow temperature', 1900.0), heff), bb.inputs['Temperature'])
    em = c.math('MULTIPLY', c.p('Emission strength', 2.5), c.math('MULTIPLY', heff, heff))
    base = c.mixc(c.p('Crust colour', (0.025, 0.018, 0.016)), (0.4, 0.04, 0.0, 1.0), heff)
    nrm = c.bump(c.p('Crust bump', 0.6), c.math('MULTIPLY', nz, c.math('SUBTRACT', 1.0, heff)), 0.03)
    pr = c.prin({'Base Color': base, 'Roughness': c.mixf(0.85, 0.35, heff), 'Specular IOR Level': 0.3, 'IOR': 1.45,
                 'Emission Color': bb.outputs[0], 'Emission Strength': em})
    c.set(pr.inputs['Normal'], nrm)
    c.finish(surface=pr.outputs[0])
    return 'emissive'


# ----------------------------------------------------------------------------- volumes
def _vol_principled(c, color, dens, aniso, noise=0.0, noise_scale=5.0, emis=0.0, emis_col=None, bb=0.0, temp_default=1500.0, absorb_col=None,
                    use_noise_name='Breakup'):
    col = c.fluid_color(c.p('Color', color))
    dn = c.p('Density', dens)
    an = c.p('Anisotropy', aniso)
    dm = c.density_mult()
    f = c.falloff()
    d = c.math('MULTIPLY', c.math('MULTIPLY', dn, dm), f)
    if noise > 0:
        nz = c.noise(c.p('Noise scale', noise_scale), 6.0, 0.6)
        amt = c.p(use_noise_name, noise)
        d = c.math('MULTIPLY', d, c.math('ADD', c.math('SUBTRACT', 1.0, amt), c.math('MULTIPLY', nz, c.math('MULTIPLY', amt, 2.0))))
    kv = dict(Color=col, Density=d, Anisotropy=an)
    if absorb_col is not None:
        kv['Absorption Color'] = c.fluid_color(c.p('Absorption colour', absorb_col), "Use fluid colour (absorption)")
    else:
        kv['Absorption Color'] = col
    if emis:
        kv['Emission Strength'] = c.p('Emission strength', emis)
        kv['Emission Color'] = c.fluid_color(c.p('Emission colour', emis_col or color), "Use fluid colour (emission)")
    if bb:
        t = c.fluid_temp(temp_default)
        ts = c.p('Temperature scale', 1.0)
        kv['Blackbody Intensity'] = c.p('Blackbody intensity', bb)
        kv['Temperature'] = c.math('MULTIPLY', t, ts)
    pv = c.volume_pr(**kv)
    if c.grid:
        for k, v in (('Density Attribute', 'density'), ('Temperature Attribute', 'temperature')):
            try:
                inp(pv, k).default_value = v
            except Exception:
                pass
    c.finish(volume=pv.outputs[0])
    return 'volume'


def b_smoke(c):
    return _vol_principled(c, (0.55, 0.55, 0.56), 3.0, 0.25, noise=0.5, noise_scale=4.0, bb=0.25, temp_default=300.0)


def b_steam(c):
    return _vol_principled(c, (0.96, 0.97, 0.98), 0.8, 0.5, noise=0.7, noise_scale=3.0)


def b_soot(c):
    return _vol_principled(c, (0.02, 0.02, 0.02), 4.0, 0.0, noise=0.45, noise_scale=6.0, absorb_col=(0.0, 0.0, 0.0))


def b_flame(c):
    return _vol_principled(c, (1.0, 0.45, 0.1), 0.6, 0.0, noise=0.4, noise_scale=7.0, emis=0.0, bb=2.5, temp_default=1600.0)


def b_plasma(c):
    return _vol_principled(c, (0.45, 0.35, 1.0), 0.5, 0.2, noise=0.5, noise_scale=5.0, emis=1.5, emis_col=(0.45, 0.4, 1.0), absorb_col=(0.9, 0.9, 1.0))


def b_gas_cloud(c):
    return _vol_principled(c, (0.7, 0.75, 0.95), 0.6, 0.45, noise=0.6, noise_scale=2.5, emis=0.35, emis_col=(0.6, 0.7, 1.0), absorb_col=(0.9, 0.9, 0.9))


# key: (label, builder, kind, description, engine tips)
PRESETS = {
    'WATER': ("Water", b_water, 'SURFACE', "Clear water: refraction (IOR 1.333), absorption + faint scatter volume, optional ripples",
              "EEVEE: Raytracing on, material Raytraced Refraction; Cycles: enable caustics on a light + receiver floor."),
    'SEA_WATER': ("Sea water", b_sea_water, 'SURFACE', "Green-blue deep water: stronger absorption + scatter, wave bump", "As water."),
    'GLASS': ("Glass", b_glass, 'SURFACE', "Clear glass, IOR 1.5, faint green edge absorption", "As water; a thin Thin Wall option is on the Principled node."),
    'GLASS_THIN': ("Glass (thin wall)", b_glass_thin, 'SURFACE', "Hollow vessel glass: thin-walled shell with the liquid inside it", "As glass."),
    'OIL': ("Oil", b_oil, 'SURFACE', "Amber oil: IOR 1.47, absorption by colour, slightly rough", "As water."),
    'HONEY': ("Honey", b_honey, 'SURFACE', "Honey: IOR 1.49, strong red-orange absorption, scattering, soft coat", "As water."),
    'SYRUP': ("Syrup", b_syrup, 'SURFACE', "Dark syrup / molasses: very dense absorption, strong coat", "As water."),
    'MILK': ("Milk / cream", b_milk, 'SURFACE', "Milk and cream: subsurface scattering, soft coat", "EEVEE: Subsurface works; enable Thickness from Shadow for thin parts."),
    'WAX': ("Wax", b_wax, 'SURFACE', "Wax: subsurface, glossy coat, warms and glows with temperature", "As milk."),
    'LAVA': ("Lava / magma", b_lava, 'SURFACE', "Molten rock: blackbody emission from temperature with a noise-driven cooled crust", "Emission lights the scene in Cycles; add Bloom-like glare in compositing."),
    'MERCURY': ("Mercury", b_mercury, 'SURFACE', "Liquid mercury: perfect smooth metal", "Needs an environment / HDRI to reflect."),
    'METAL': ("Molten metal", b_metal, 'SURFACE', "Metal that glows by blackbody when hot (copper default)", "Needs an environment to reflect."),
    'SLIME': ("Slime / goo", b_slime, 'SURFACE', "Goo: subsurface, glassy coat, bumpy", "As milk."),
    'BLOOD': ("Blood", b_blood, 'SURFACE', "Blood-like fluid: red subsurface under a wet coat, thin red absorption", "As milk."),
    'ICE': ("Ice", b_ice, 'SURFACE', "Ice: IOR 1.31, frosted roughness, faint cyan absorption + scatter", "As water."),
    'CRYSTAL': ("Crystal", b_crystal, 'SURFACE', "Faceted gem: high IOR, per-cell facet normals, absorption (dispersion on 5.3+)", "As water."),
    'SMOKE': ("Smoke", b_smoke, 'VOLUME', "Grey smoke: Principled Volume, noise breakup, faint blackbody glow while hot", "EEVEE: volume shaders render inside the object's mesh; Cycles: volume step rate."),
    'STEAM': ("Steam", b_steam, 'VOLUME', "White steam: forward scattering, strong breakup", "As smoke."),
    'SOOT': ("Soot", b_soot, 'VOLUME', "Black soot: absorbing, no scattering", "As smoke."),
    'FLAME': ("Flame", b_flame, 'VOLUME', "Fire: blackbody emission from temperature", "Cycles: set volume sampling to Multiple Importance."),
    'PLASMA': ("Plasma", b_plasma, 'VOLUME', "Glowing ionised gas: emission-driven violet plasma with scatter", "As flame."),
    'GAS_CLOUD': ("Gas cloud", b_gas_cloud, 'VOLUME', "Soft nebular gas cloud: scatter + faint emission", "As smoke."),
}
ORDER = list(PRESETS)
PRESET_ITEMS = [(k, v[0], v[3]) for k, v in PRESETS.items()]

# Fluid-mode preset -> default shader preset
FLUID_DEFAULT = {'WATER': 'WATER', 'VEG_OIL': 'OIL', 'MINERAL_OIL': 'OIL', 'LAVA_WAX': 'WAX', 'HONEY': 'HONEY', 'AIR': 'GAS_CLOUD',
                 'HELIUM': 'GAS_CLOUD', 'CO2': 'SMOKE', 'CUSTOM': 'WATER'}
# Fluid defaults carry their own params
_FLUID_OVER = {'MINERAL_OIL': {'Color': (0.92, 0.88, 0.6), 'Absorption density': 2.0}, 'CO2': {'Color': (0.5, 0.52, 0.58), 'Density': 0.6}}


def shader_for_fluid(fl):
    """Shader preset key for a fluid (its own choice, or the default of its fluid preset)."""
    s = getattr(fl, 'shader', 'AUTO')
    if s in PRESETS:
        return s
    return FLUID_DEFAULT.get(getattr(fl, 'preset', 'CUSTOM'), 'WATER')


# ============================================================================= material factory
def material_name(key, preview=False, particle=False, grid=False):
    return PREFIX + PRESETS[key][0] + (" (preview)" if preview else "") + (" (particles)" if particle else "") + (" (grid)" if grid else "")


def apply_settings(mat, key, mode, preview=False, particle=False):
    """Render-engine settings of the material: EEVEE refraction / shadow / volume options, Cycles volume sampling."""
    kind = PRESETS[key][2]
    sets = {}
    if mode == 'refractive' and not preview and not particle:
        sets.update(surface_render_method='DITHERED', use_raytrace_refraction=True, use_transparent_shadow=True, use_screen_refraction=True)
    elif mode in ('blend', 'refractive') or particle:
        sets.update(surface_render_method='BLENDED' if (preview or particle) else 'DITHERED', use_transparent_shadow=True)
    elif mode == 'sss':
        sets.update(surface_render_method='DITHERED', use_thickness_from_shadow=True)
    elif mode in ('emissive', 'metal'):
        sets.update(surface_render_method='DITHERED')
    if kind == 'VOLUME':
        sets.update(volume_intersection_method='ACCURATE')
    for k, v in sets.items():
        try:
            setattr(mat, k, v)
        except Exception:
            pass
    try:
        mat.cycles.volume_sampling = 'MULTIPLE_IMPORTANCE' if key in ('FLAME', 'PLASMA', 'GAS_CLOUD') else 'DISTANCE'
        mat.cycles.volume_interpolation = 'CUBIC' if kind == 'VOLUME' else 'LINEAR'
        mat.cycles.emission_sampling = 'AUTO'
    except Exception:
        pass
    # viewport colour (solid mode / material preview fallback)
    for nm in ('Color',):
        n = mat.node_tree.nodes.get('TS_' + nm)
        if n is not None:
            c = tuple(n.outputs[0].default_value)
            mat.diffuse_color = (c[0], c[1], c[2], 0.35 if mode in ('refractive', 'blend', 'volume') else 1.0)


def build(key, name=None, preview=False, particle=False, grid=False, strict=False, reuse=True, **over):
    """Build (or reuse) the material of a preset.  ``over`` sets parameters by name (``Absorption_density=3``); a reused material keeps the
    user's edits unless parameters are given.  Returns the material, tagged with ``ts_preset`` / ``ts_variant``."""
    label, fn, kind, desc, tips = PRESETS[key]
    name = name or material_name(key, preview, particle, grid)
    mat = bpy.data.materials.get(name)
    if mat is not None and reuse and mat.get('ts_preset') == key and mat.node_tree and mat.node_tree.nodes:
        if over:
            set_params(mat, **over)
        return mat
    if mat is None:
        mat = bpy.data.materials.new(name)
    full_over = dict(over)
    c = Ctx(mat, full_over, preview, particle, grid, strict)
    mode = fn(c)
    apply_settings(mat, key, mode, preview, particle)
    mat['ts_preset'] = key
    mat['ts_variant'] = "preview" if preview else ("particles" if particle else ("grid" if grid else "full"))
    mat['ts_mode'] = mode
    mat['ts_unresolved'] = ", ".join(c.unresolved)
    return mat


def set_params(mat, **kv):
    """Set parameter nodes of a preset material by name (underscores = spaces, case-insensitive).  Returns the names not found."""
    missing = []
    if mat is None or mat.node_tree is None:
        return list(kv)
    index = {n.name[3:].lower(): n for n in mat.node_tree.nodes if n.name.startswith('TS_')}
    for k, v in kv.items():
        n = index.get(k.lower().replace('_', ' '))
        if n is None:
            missing.append(k)
            continue
        cur = n.outputs[0].default_value
        if isinstance(cur, float):
            n.outputs[0].default_value = float(v)
        else:
            vv = tuple(v)
            n.outputs[0].default_value = vv if len(vv) == 4 else (vv[0], vv[1], vv[2], 1.0)
    return missing


def params(mat):
    """The parameter nodes (TS_*) of a preset material, in creation order."""
    if mat is None or mat.node_tree is None:
        return []
    return [n for n in mat.node_tree.nodes if n.name.startswith('TS_')]


def preset_of(mat):
    return mat.get('ts_preset') if mat is not None else None


def is_refractive(key):
    return PRESETS[key][1] in (b_water, b_sea_water, b_glass, b_glass_thin, b_oil, b_honey, b_syrup, b_ice, b_crystal)


# ============================================================================= assignment
def _material_target(obj):
    d = obj.data
    return d if (d is not None and hasattr(d, 'materials')) else None


def assign(obj, mat):
    """Put the material on every slot of an object (one slot is created when it has none).  Works on meshes, curves, text, metaballs,
    volumes and point clouds.  Returns False for objects without material slots (empties, lights)."""
    d = _material_target(obj)
    if d is None:
        return False
    if len(d.materials) == 0:
        d.materials.append(mat)
    else:
        for i in range(len(d.materials)):
            d.materials[i] = mat
    return True


def caustics_hints(scene, casters=(), receivers=(), lights=()):
    """Cycles shadow caustics (3.2+, MNEE): casters = refractive objects, receivers = floors / walls, lights = the (small) lights to use."""
    try:
        scene.cycles.caustics_refractive = True
        scene.cycles.caustics_reflective = True
    except Exception:
        pass
    for o in casters:
        try:
            o.cycles.is_caustics_caster = True
        except Exception:
            pass
    for o in receivers:
        try:
            o.cycles.is_caustics_receiver = True
        except Exception:
            pass
    for o in lights:
        try:
            o.data.cycles.is_caustics_light = True
        except Exception:
            pass


def assign_preset(objs, key, preview=False, caustics=False, scene=None, name=None, unique=False, **over):
    """Assign preset ``key`` to objects.  Volume objects get the grid variant, mesh liquids the full one.  Returns (material, n assigned)."""
    objs = list(objs)
    grid = PRESETS[key][2] == 'VOLUME' and any(o.type == 'VOLUME' for o in objs)
    if unique:
        mat = build(key, name=name, preview=preview, grid=grid, reuse=False, **over)
    else:
        mat = build(key, name=name, preview=preview, grid=grid, **over)
    n = 0
    casters = []
    for o in objs:
        if assign(o, mat):
            n += 1
            if is_refractive(key):
                casters.append(o)
    if caustics and scene is not None and casters:
        caustics_hints(scene, casters=casters)
    return mat, n


def fluid_variant(fl):
    """Material variant a fluid needs: SURFACE render mode -> the full material; points / instances / raymarch / splat -> the particle variant."""
    return fl.render_mode != 'SURFACE'


def fluid_material(fl, preview=False, surface=None):
    """Material of a fluid (for its points object, or its surface mesh when ``surface``): preset by fluid, fm_color / temperature / density on."""
    key = shader_for_fluid(fl)
    particle = (not surface) and PRESETS[key][2] == 'SURFACE' and is_refractive(key)
    name = "%s%s [%s]" % (material_name(key, preview, particle), " surface" if surface else "", fl.name)
    base = material_name(key, preview, particle)
    mat = bpy.data.materials.get(name)
    if mat is None or mat.get('ts_preset') != key:
        # per-fluid copy so the user can tweak one fluid's look
        mat = build(key, name=name, preview=preview, particle=particle, reuse=False, **_FLUID_OVER.get(getattr(fl, 'preset', ''), {}))
        set_params(mat, Use_fluid_temperature=1.0)
        if key == 'WAX':
            set_params(mat, Cold_temperature=float(fl.t_lo), Hot_temperature=float(fl.t_hi))
        if 'color' in dir(fl) and mat.node_tree.nodes.get('TS_Color') is not None:
            set_params(mat, Color=tuple(fl.color))
        # subsurface radii are in scene units: scale them to the size of the fluid's blobs (about 20 particles across)
        if not surface and mat.get('ts_mode') == 'sss':
            set_params(mat, Subsurface=0.0)           # random-walk subsurface inside millimetre spheres is pure noise: diffuse + coat instead
        n = mat.node_tree.nodes.get('TS_Subsurface scale')
        dx = _solver_dx()
        an = mat.node_tree.nodes.get('TS_Absorption density')
        if an is not None and dx and is_refractive(key):   # densities are per metre: a 10 cm vessel needs ~10x more than a 1 m tank
            an.outputs[0].default_value = an.outputs[0].default_value / min(1.0, 20.0 * dx)
        if surface and mat.get('ts_mode') == 'refractive':
            # attributes are not defined inside a volume: the per-vertex colour tints the surface, the absorption stays a mild depth cue
            set_params(mat, Surface_tint=0.75)
            if an is not None:
                an.outputs[0].default_value *= 0.35
        if n is not None and dx:
            n.outputs[0].default_value = min(n.outputs[0].default_value, n.outputs[0].default_value * 20.0 * dx)
    # fm_color is only meaningful when the fluids mix (otherwise the attribute is a white placeholder): switch the colour input with it
    mixing = 1.0 if (surface or _solver_mixing()) else 0.0   # surface meshes always carry their vertex colours
    set_params(mat, Use_fluid_colour=mixing, **{'Use fluid colour (absorption)': mixing, 'Use fluid colour (emission)': mixing})
    return mat


def _solver_mixing():
    try:
        from . import fluidmode_ops as FO
        return bool(getattr(FO.RT.solver, 'mixing', False)) if FO.RT.solver is not None else False
    except Exception:
        return False


def _solver_dx():
    try:
        from . import fluidmode_ops as FO
        return float(FO.RT.solver.dx) if FO.RT.solver is not None else 0.0
    except Exception:
        return 0.0


def apply_fluid_shaders(scene, preview=None):
    """Assign the shader preset of every fluid to its render objects (points object's render group, surface mesh).  Safe to call repeatedly."""
    from . import gn
    fm = getattr(scene, 'arc_flow_fm', None)
    if fm is None:
        return 0
    if preview is None:
        preview = bool(getattr(getattr(scene, 'arc_flow_sh', None), 'preview', False))
    n = 0
    for fl in fm.fluids:
        if getattr(fl, 'shader', 'AUTO') == 'OFF':
            continue
        o = fl.points_obj
        if o is not None:
            mr = o.modifiers.get("AF Render")
            if mr is not None and mr.node_group is not None:
                gn.set_group_material(mr.node_group, fluid_material(fl, preview))
                n += 1
        so = bpy.data.objects.get("AFFMS_%s" % fl.name)
        if so is not None and so.data is not None:
            m = fluid_material(fl, preview, surface=True)
            so.data.materials.clear()
            so.data.materials.append(m)
            n += 1
    n += apply_merged_surface(scene, preview)
    return n


def apply_merged_surface(scene, preview=False):
    """The merged liquid surface (AFFMS_Liquids) takes the shader of the first liquid that renders as a surface (per vertex colours come from fm_color)."""
    so = bpy.data.objects.get("AFFMS_Liquids")
    fm = getattr(scene, 'arc_flow_fm', None)
    if so is None or so.data is None or fm is None:
        return 0
    for fl in fm.fluids:
        if fl.render_mode == 'SURFACE' and getattr(fl, 'shader', 'AUTO') != 'OFF':
            m = fluid_material(fl, preview, surface=True)
            so.data.materials.clear()
            so.data.materials.append(m)
            return 1
    return 0


def surface_material(fl, scene=None):
    """Material for a fluid's surface mesh (hook for the surface builder); None = keep the composition material."""
    if getattr(fl, 'shader', 'AUTO') == 'OFF':
        return None
    prev = bool(getattr(getattr(scene, 'arc_flow_sh', None), 'preview', False)) if scene is not None else False
    return fluid_material(fl, prev, surface=True)
