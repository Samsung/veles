#!/bin/bash
n_bframes=8
n_refs=6
n_subq=7
s_flags=+loop+mv4
s_flags2=+mbtree+wpred+mixed_refs+dct8x8
crf=26
fps=30

ffmpeg -y -r $fps -f image2pipe -vcodec ppm -i $1 -vcodec libx264 -crf $crf -bf $n_bframes -subq $n_subq -cmp 256 -refs $n_refs -qmin 10 -qmax 51 -qdiff 4 -coder 1 -me_method umh -me_range 16 -trellis 1 -flags $s_flags -partitions parti4x4+parti8x8+partp4x4+partp8x8+partb8x8 -g 250 -keyint_min 25 -sc_threshold 40 -i_qfactor 0.71 -threads 16 $2
