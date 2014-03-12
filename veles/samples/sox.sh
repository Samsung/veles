#!/bin/sh
in=$1
start=$2
tmp1=/tmp/tmp1.wav
tmp2=/tmp/tmp2.wav
rm -f $tmp1
sox "$in" $tmp1 trim $start 30
rm -f $tmp2
sox $tmp1 $tmp2 remix 1
#sox $tmp1 -c 1 $tmp2
rm -f $tmp1
sox $tmp2 -r 22050 -G $tmp1
rm -f $tmp2
./forward_gtzan.py -f=/data/veles/music/features.xml -snapshot=/data/veles/music/GTZAN/gtzan_1000_500_10_28.88pt_Wb.pickle -file=$tmp1 -shift_size=50 -name="$in" -graphics=1
