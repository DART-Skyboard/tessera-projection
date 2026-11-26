#!/bin/bash
# usage: assemble_hair_ariel.sh FRAMES_DIR OUT.mp4 [STILLS_DIR]
# One UI screenshot per 2 sim frames (12 grabs per second of scene time); each grab is held for 2 output frames at 24 fps, so the clip
# plays 30 s of scene time in 30 s of video (the hair moves in 12 Hz steps).  Stills: ON with bodies shown, OFF, ON again.
set -e
F=$1; O=$2; S=${3:-$(dirname $O)}
ffmpeg -y -loglevel error -framerate 12 -i $F/f%05d.jpg -vf "fps=24,format=yuv420p" -c:v libx264 -crf 20 -preset veryfast $O
python3 - "$F" "$S" <<'PY'
import json, shutil, sys
F, S = sys.argv[1:3]
m = json.load(open(F + '/manifest.json'))
def pick(pred, frac):
    c = [e for e in m if pred(e)]
    return c[int(len(c) * frac)]['file'] if c else None
sel = [pick(lambda e: e['collision'] and e['bodies'] and e['video_t'] < 14.75, 0.8),
       pick(lambda e: not e['collision'], 0.6),
       pick(lambda e: e['collision'] and e['bodies'] and e['video_t'] > 18.75, 0.6)]
for i, f in enumerate(sel, 1):
    if f:
        shutil.copy(F + '/' + f, '%s/hair_ariel_ui_still%d.jpg' % (S, i))
print(sel)
PY
