#!/bin/bash
n_ref=5
n_bframes=5
n_subme=3
crf=23
fps=30
keyint=300
preset=slow
tmp=/tmp/Veles.265

# Via x265
/opt/bin/ffmpeg -r $fps -f image2pipe -vcodec ppm -i $1 -f yuv4mpegpipe -pix_fmt yuv420p - | x265 --y4m --preset $preset --crf $crf --bframes $n_bframes --ref $n_ref --subme $n_subme --keyint $keyint --fps $fps - -o $tmp
/opt/bin/ffmpeg -i $tmp -c copy $2
# With audio
#/opt/bin/ffmpeg -i $tmp -i INPUT_AUDIO -map 0 -map 1:a -c copy $2
