"""Every icon name used in the add-on exists in this Blender's UILayout icon enum (also draws the panels' icon strings via a source scan).
blender --factory-startup -b --python tests/test_icons.py"""
import os
import re
import sys
import bpy
HERE = os.path.dirname(os.path.abspath(__file__))
valid = {e.identifier for e in bpy.types.UILayout.bl_rna.functions['label'].parameters['icon'].enum_items}
bad = []
n = 0
root = os.path.join(HERE, '..', 'arc_flow')
for dp, _d, fs in os.walk(root):
    for f in fs:
        if not f.endswith('.py'):
            continue
        src = open(os.path.join(dp, f)).read()
        for m in re.finditer(r"""icon\s*=\s*['"]([A-Z0-9_]+)['"]""", src):
            n += 1
            if m.group(1) not in valid:
                bad.append((f, src.count('\n', 0, m.start()) + 1, m.group(1)))
        for m in re.finditer(r"""\(\s*['"][A-Z0-9_]+['"]\s*,\s*['"][^'"]*['"]\s*,\s*['"][^'"]*['"]\s*,\s*['"]([A-Z0-9_]+)['"]\s*,\s*\d+\s*\)""", src):
            n += 1
            if m.group(1) not in valid:
                bad.append((f, src.count('\n', 0, m.start()) + 1, m.group(1)))
print("%d icon names checked on Blender %s" % (n, bpy.app.version_string))
for b in bad:
    print("FAIL invalid icon %s:%d %s" % b)
print("PASS all icon names valid" if not bad else "FAILURES: %d" % len(bad))
if not bad:
    print("FAILURES: 0")
sys.exit(1 if bad else 0)
