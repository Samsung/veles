#!/bin/sh -e

mkdir -p projects/common/node_modules
mkdir -p projects/common/src/libs

if ! which npm > /dev/null; then
  echo "You need node.js to build: sudo apt-get install nodejs" 1>&2
  exit 1
fi

rm -rf web/dist
cd projects/common
ln -s ../core/package.json package.json
npm install
rm package.json

cd ../core
npm install
npm run gulp

cd ../forge
npm install
npm run gulp
