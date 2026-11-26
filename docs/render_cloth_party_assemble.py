"""Assemble the stepped UI capture (WORK/ui_frames/manifest.json) into an mp4 at 24 fps.

    python3 docs/render_cloth_party_assemble.py WORKDIR OUT.mp4 [fps=24]

Every captured frame is held for (frame gap / fps) seconds, i.e. frames grabbed every 2nd solver frame are shown for 2 output frames
(a stepped / held look; no interpolation).  Also writes 3 stills and a 5-frame contact sheet next to the mp4.
"""
import json
import os
import subprocess
import sys

work, out = sys.argv[1], sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 24
hold = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0   # slow-motion factor (each grab is shown hold x longer)
d = os.path.join(work, 'ui_frames')
man = json.load(open(os.path.join(d, 'manifest.json')))
man.sort(key=lambda e: e['frame'])
lst = os.path.join(work, 'concat.txt')
with open(lst, 'w') as fh:
    for i, e in enumerate(man):
        nxt = man[i + 1]['frame'] if i + 1 < len(man) else e['frame'] + 2
        fh.write("file '%s'\nduration %.6f\n" % (os.path.join(d, e['file']), hold * (nxt - e['frame']) / fps))
    fh.write("file '%s'\n" % os.path.join(d, man[-1]['file']))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-vf', 'fps=%d,format=yuv420p' % fps,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-movflags', '+faststart', out], check=True)
base = os.path.splitext(out)[0]
n = len(man)
pick = [man[int(k * (n - 1) / 4)] for k in range(5)]
for k, e in enumerate((man[int(n * 0.15)], man[int(n * 0.5)], man[int(n * 0.85)])):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', os.path.join(d, e['file']), base + '_still%d.png' % (k + 1)], check=True)
ins = []
for e in pick:
    ins += ['-i', os.path.join(d, e['file'])]
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + ins + ['-filter_complex',
                ''.join('[%d:v]scale=640:-1[s%d];' % (i, i) for i in range(5)) + ''.join('[s%d]' % i for i in range(5)) +
                'xstack=inputs=5:layout=0_0|640_0|1280_0|0_360|640_360', '-frames:v', '1', base + '_contact.png'], check=True)
print("wrote", out, os.path.getsize(out))
