#!/bin/sh -e

if [ -z "$1" ]; then
  echo "You must specify the archive destination directory." 1>&2
  exit 1
fi

path=$(readlink -e $1)
git archive --format=tar HEAD -o $path/Veles.tar
cd veles/znicz
git archive --format=tar --prefix veles/znicz/ HEAD -o $path/Znicz.tar
cd $path
tar --concatenate --file Veles.tar Znicz.tar
rm Znicz.tar
xz Veles.tar
