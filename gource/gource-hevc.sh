#!/bin/bash
n_ref=5
n_bframes=5
n_subme=3
crf=23
fps=60
res=1920x1080
keyint=480
sar=16:9

ffmpeg -r $fps -f image2pipe -vcodec ppm -i $1 -f yuv4mpegpipe -pix_fmt yuv420p - | x265 --y4m --preset slow --sar $sar --crf $crf --bframes $n_bframes --ref $n_ref --subme $n_subme --keyint $keyint --input-res $res --fps $fps - -o $2
