#!/bin/sh -e

PYVER=3.4.1
COMPRESSION=xz

if [ -z "$1" ]; then
  echo "You must specify either \"pre\" or \"post\" command" 1>&2
  exit 1
fi

root=$(readlink -f $(dirname $(readlink -f $0))/..)
path=$root/deploy
cpus=$(getconf _NPROCESSORS_ONLN)

do_pre() {
  echo "Running git archive..."
  cd $root
  git archive --format=tar HEAD -o $path/Veles.tar
  cd $root/veles/znicz
  git archive --format=tar --prefix veles/znicz/ HEAD -o $path/Znicz.tar
  cd $root/deploy/pyenv
  git archive --format=tar --prefix deploy/pyenv/ HEAD -o $path/pyenv.tar
  cd $path
  echo "Merging archives..."
  tar --concatenate --file Veles.tar Znicz.tar
  tar --concatenate --file Veles.tar pyenv.tar
  rm Znicz.tar pyenv.tar
  echo "Compressing..."
  rm -f Veles.tar.$COMPRESSION
  $COMPRESSION Veles.tar
  echo "$path/Veles.tar.$COMPRESSION is ready"
}

do_post() {
  cd $path
  . ./init-pyenv
  versions=$(pyenv versions | grep $PYVER)
  if [ -z "$versions" ]; then
    pyenv install $PYVER
  fi
  pyenv local $PYVER

  vroot=$path/pyenv/versions/$PYVER
  if [ ! -e $vroot/lib/libsodium.so ]; then
    git clone https://github.com/jedisct1/libsodium
    cd libsodium && mkdir build
    ./autogen.sh && cd build
    ../configure --prefix=$vroot --disable-static
    make -j$cpus && make install
    cd ../.. && rm -rf libsodium
  fi

  if [ ! -e $vroot/lib/libpgm.so ]; then
    svn checkout http://openpgm.googlecode.com/svn/trunk/ openpgm
    cd openpgm/openpgm/pgm && mkdir build
    patch if.c < ../../../openpgm.patch
    autoreconf -i -f && cd build
    ../configure --prefix=$vroot --disable-static
    make -j$cpus && make install
    cd ../../../.. && rm -rf openpgm
  fi

  if [ ! -e $vroot/lib/libzmq.so.4 ]; then
    git clone https://github.com/vmarkovtsev/libzmq.git
    cd libzmq && mkdir build
    ./autogen.sh && cd build
    ../configure --prefix=$vroot --disable-static --without-documentation --with-system-pgm --with-libsodium-include-dir=$vroot/include --with-libsodium-lib-dir=$vroot/lib PKG_CONFIG_PATH=$vroot/lib/pkgconfig PKG_CONFIG_LIBDIR=$vroot/lib
    make -j$cpus && make install
    cd ../.. && rm -rf libzmq
  fi

   pip3 install cython
   pip3 install git+https://github.com/vmarkovtsev/twisted.git
   PKG_CONFIG_PATH=$vroot/lib/pkgconfig pip3 install -r $root/requirements.txt
}

case "$1" in
  "pre"):
     do_pre
     ;;
  "post"):
     do_post
     ;;
esac
