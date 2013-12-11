#!/bin/bash
base=$3/
path=$1/$2
path=${path/$base/}
sed_path=${path//\//\\\/}
if [[ $path != "libVeles/libarchive" && $path != "libVeles/zlib" && $path != "libSoundFeatureExtraction/DSPFilters" && $path != "Znicz/libZnicz/simd" ]]; then
git log --pretty=format:user:%aN%n%ct --reverse --raw --encoding=UTF-8 --no-renames | sed -r -e "s/(\.\.\. \w\s+)/\1$sed_path\//g" -e 's/:[^A-Z]+(\w)\s+(.*)/|\1|\2/g'
fi
