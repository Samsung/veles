#!/bin/bash

git submodule --quiet foreach --recursive '/home/markhor/Development/Veles/gource-logger.sh $toplevel $path' > history.log
git log --pretty=format:user:%aN%n%ct --reverse --raw --encoding=UTF-8 --no-renames | sed -r -e "s/(\.\.\. \w\s+)/\1$sed_path\//g" -e 's/:[^A-Z]+(\w)\s+(.*)/|\1|\2/g' >> history.log
./gource-log-fix.py history.log
rm history.log
echo "gource --load-config veles.gource --log-format custom -s 0.5 --highlight-all-users"
