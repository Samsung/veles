#!/bin/sh -e

mkdir -p projects/common/node_modules
mkdir -p projects/common/src/libs

if ! which npm > /dev/null; then
  echo "You need node.js to build: sudo apt-get install nodejs" 1>&2
  exit 1
fi

rm -rf web/dist
cd projects/common
rm -f package.json
ln -s ../core/package.json package.json
npm install
rm package.json

cd ../core
npm install
if ! npm run gulp; then
  # Sometimes it fails for the first time; the error is different each time
  # and it looks like a memory corruption
  npm run gulp nuke
  npm run gulp
fi

cd ../forge
npm install
npm run gulp

cd ../bboxer
npm install
npm run gulp
