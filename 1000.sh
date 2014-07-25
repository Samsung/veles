#!/bin/sh
echo Folder: $1
dev=$2
#nodes=markovtsevu64/1:0,kuznetsovu64/0:0,seresovu64/0:0,kazantsevu64/0:0,golovizinu64/1:0,lpod/1:0,seninu64/1:0,smaug/0:0-1,smaug/1:0-1,intern23/0:0
nodes=lpod/1:0,
script="scripts/velescli.py -d $dev -l 0.0.0.0:5000 -n $nodes --manhole --debug NNSnapshotter,Snapshotter -f /tmp/1000.log -p '' "
snapshot="-w /data/veles/datasets/imagenet/snapshots/"$1"/imagenet_ae_"$1"_parallel_current.3.pickle"
workflow="veles/znicz/tests/research/imagenet/imagenet_ae.py veles/znicz/tests/research/imagenet/imagenet_ae_temp_paral_config.py"
params="root.loader.year=\""$1"\" root.common.plotters_disabled=True"
cmd0=""$script" "$workflow" "$params
cmd1=""$script" "$snapshot" "$workflow" "$params
delay=180

echo "Will execute:"
echo $cmd0
if ! $cmd0; then
  echo "Failed, exit"
  exit
fi

echo "Will train next layer after "$delay" seconds"
sleep $delay
echo "Will execute:"
echo $cmd1
$cmd1
echo "Will train next layer after "$delay" seconds"
sleep $delay
$cmd1
echo "Will train next layer after "$delay" seconds"
sleep $delay
$cmd1
echo "Will train next layer after "$delay" seconds"
sleep $delay
$cmd1

echo "Will train classifier:"
echo "Will train next layer after "$delay" seconds"
sleep $delay
$cmd1

echo "Will fine-tune:"
echo "Will train next layer after "$delay" seconds"
sleep $delay
$cmd1
