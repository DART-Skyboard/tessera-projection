"""Per-frame layer-crossing table for a solved garment group (no bpy; plain python3).

    python3 docs/cloth_crossings.py CACHE.npz [every=12] [out.md]

For every `every`-th frame: vertices of member P that sit on the wrong (non-rest) side of the nearest face of member Q
they project onto, for every ordered member pair, plus the deepest one (mm).  Independent of the solver: brute-force
matrix test on the stored positions (core.cloth_selfcol.SelfCollider.wrong_side).  Prints min / max / worst frame.
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('cloth_selfcol', os.path.join(HERE, '..', 'arc_flow', 'core', 'cloth_selfcol.py'))
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)


def table(cache, every=12, names=None, verbose=True):
    z = np.load(cache)
    X, rest, tris, mem = z['x'], z['rest'], z['tris'], z['members']
    sc = SC.SelfCollider(tris, _edges(tris), len(rest))
    sc.set_rest(rest)
    k = int(mem.max()) + 1
    names = names or ["m%d" % i for i in range(k)]
    pairs = [(a, b) for a in range(k) for b in range(k) if a != b]
    rows = []
    for f in range(0, len(X), every):
        ws = sc.wrong_side(X[f].astype(float), labels=mem)
        rows.append((f + 1, {p: ws[p] for p in pairs}))
        if verbose:
            print(f + 1, {"%s>%s" % (names[a], names[b]): v[0] for (a, b), v in rows[-1][1].items()}, flush=True)
    return rows, pairs, names


def _edges(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def summary(rows, pairs, names):
    out = ["| frame | " + " | ".join("%s -> %s" % (names[a], names[b]) for a, b in pairs) + " | total |",
           "|---|" + "---|" * (len(pairs) + 1)]
    for f, d in rows:
        out.append("| %d | " % f + " | ".join("%d (%.1f mm)" % (d[p][0], 1000 * d[p][1]) if d[p][0] else "0" for p in pairs)
                   + " | %d |" % sum(d[p][0] for p in pairs))
    tot = [sum(d[p][0] for p in pairs) for _f, d in rows]
    w = int(np.argmax(tot))
    out.append("")
    out.append("total wrong-side vertices over %d sampled frames: min %d, max %d (frame %d), mean %.1f"
               % (len(rows), min(tot), max(tot), rows[w][0], float(np.mean(tot))))
    for p in pairs:
        c = [d[p][0] for _f, d in rows]
        dep = max(d[p][1] for _f, d in rows)
        out.append("- %s -> %s: min %d, max %d (frame %d), deepest %.1f mm" % (names[p[0]], names[p[1]], min(c), max(c),
                                                                          rows[int(np.argmax(c))][0], 1000 * dep))
    return "\n".join(out)


if __name__ == '__main__':
    cache = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    outp = sys.argv[3] if len(sys.argv) > 3 else None
    nm = ["Bodice+Skirt", "Overskirt", "Train"]
    rows, pairs, names = table(cache, every, nm)
    txt = summary(rows, pairs, names)
    print(txt)
    if outp:
        open(outp, 'w').write(txt + "\n")
