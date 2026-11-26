#!/bin/bash
# usage: assemble_hair_ui.sh FRAMES_DIR OUT.mp4 [STILLS_DIR]
# Stepped capture: one UI screenshot per 4 sim frames (6 grabs per second of scene time); each grab is held for 4 output
# frames at 24 fps, so the clip plays at real scene speed but the hair moves in steps.
set -e
F=$1; O=$2; S=${3:-$(dirname $O)}
ffmpeg -y -loglevel error -framerate 6 -i $F/f%05d.jpg -vf "fps=24,format=yuv420p" -c:v libx264 -crf 22 -preset veryfast $O
n=$(ls $F/f*.jpg | wc -l)
for i in 1 2 3; do
  k=$(( n * (i*2-1) / 7 ))
  cp $F/$(printf 'f%05d.jpg' $k) $S/hair_collision_ui_still$i.jpg
done
