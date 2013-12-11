#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
path=$(git rev-parse --show-toplevel)
git submodule --quiet foreach --recursive "$DIR/gource-logger.sh \$toplevel \$path $path" > history.log
git log --pretty=format:user:%aN%n%ct --reverse --raw --encoding=UTF-8 --no-renames | sed -r -e "s/(\.\.\. \w\s+)/\1/g" -e 's/:[^A-Z]+(\w)\s+(.*)/|\1|\2/g' >> history.log
old_path=$(pwd)
pushd ../ChannelRecognizer
git log --pretty=format:user:%aN%n%ct --reverse --raw --encoding=UTF-8 --no-renames | sed -r -e "s/(\.\.\. \w\s+)/\1ChannelRecognizer\//g" -e 's/:[^A-Z]+(\w)\s+(.*)/|\1|\2/g' >> $old_path/history.log
popd
sed -i -e 's/EBulychev/Egor Bulychev/g' -e 's/Markovtsev Vadim/Vadim Markovtsev/g' history.log
$DIR/gource-log-fix.py history.log
rm history.log
echo "gource --load-config $DIR/veles.gource --log-format custom -s 0.5 --highlight-all-users"
