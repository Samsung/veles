#!/bin/bash
n_ref=5
n_bframes=5
n_subme=3
crf=23
fps=60
res=1920x1080
keyint=480
sar=16:9
preset=slow
tmp=/tmp/Veles.265

# http://stackoverflow.com/questions/19634453/ffmpeg-how-to-generate-a-mp4-with-h-265-codec

# Via ffmpeg with libx265
# Example: ./gource-hevc Veles.ppm Veles.mov
ffmpeg -r $fps -f image2pipe -vcodec ppm -i $1 -an -x265-params crf=$crf -pix_fmt yuv420p $2
exit

# Via x265
ffmpeg -r $fps -f image2pipe -vcodec ppm -i $1 -f yuv4mpegpipe -pix_fmt yuv420p - | x265 --y4m --preset $preset --sar $sar --crf $crf --bframes $n_bframes --ref $n_ref --subme $n_subme --keyint $keyint --input-res $res --fps $fps - -o $tmp
ffmpeg -i $tmp -c copy $2
# With audio
#ffmpeg -i $tmp -i INPUT_AUDIO -map 0 -map 1:a -c copy $2
