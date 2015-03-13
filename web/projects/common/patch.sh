#!/bin/sh -e

download_patch() {
  patch_file=$1
  url=$2
  if [ ! -e $patch_file ]; then
    if which curl; then
      curl -o $patch_file $url
    elif which wget; then
      wget -O $patch_file $url
    else
      echo "Unable to download $patch_file from GitHub" 1>&2
      exit 1
    fi
  fi
  patch_file="$(readlink -e $patch_file)"
}

download_patch bin-wrapper.patch https://github.com/vmarkovtsev/bin-wrapper/commit/ce620fc4f8835daa93e069a00ad13c69003e9456.patch
find node_modules -type d -name bin-wrapper -exec patch -Nr- {}/index.js "$patch_file" \;

touch node_modules/patched
