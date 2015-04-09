#!/bin/sh -e

mkdir -p projects/common/node_modules
mkdir -p projects/common/src/libs

cd projects/core
npm run gulp

cd ../forge
npm run gulp
