#!/bin/sh -e

PYVER=3.4.3
# The following is only used in do_pre()
if [ -z "$2" ]; then
  COMPRESSION=xz
else
  COMPRESSION=$2
fi

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
  rm veles/__init__.py
  git checkout veles/__init__.py
  cd veles/znicz
  git checkout __init__.py
  cd ../..
  git archive --format=tar HEAD -o $path/Veles.tar
  tar -cf $path/package_lists.tar docs/source/manualrst_veles_user_*_packages.rst
  cd $root/veles/znicz
  git archive --format=tar --prefix veles/znicz/ HEAD -o $path/Znicz.tar
  cd $root/deploy/pyenv
  git archive --format=tar --prefix deploy/pyenv/ HEAD -o $path/pyenv.tar
  cd $root/mastodon
  git archive --format=tar --prefix mastodon/ HEAD -o $path/Mastodon.tar
  cd $root/web
  echo "Building web..."
  ./build.sh
  cd $root
  tar -cf $path/web.tar web/dist
  cd $path
  echo "Merging archives..."
  for arch in package_lists.tar Znicz.tar pyenv.tar Mastodon.tar web.tar; do
    tar --concatenate --file Veles.tar $arch
    rm $arch
  done
  echo "Compressing..."
  rm -f Veles.tar.$COMPRESSION
  $COMPRESSION Veles.tar
  echo "$path/Veles.tar.$COMPRESSION is ready"
}

check_dist() {
  major=$(lsb_release -r | cut -d : -f 2 | tr -d '\t' | cut -d . -f 1)
  if [ $major -lt $2 ]; then
    echo "$1 older than $2.x is not supported" 1>&2
    exit 1
  fi
}

debian_based_setup() {
  check_dist "$1" "$2"
  packages=$(cat "$root/$3" | tail -n +9 | sed -r -e 's/^\s+//g' -e 's/\\//g' | tr '\n' ' ')
  need_install=""
  for package in $packages; do
    if ! dpkg -l | grep -qG "ii  $package[: ]"; then
      echo "$package is not installed"
      need_install="yes"
    fi
  done
  if [ ! -z $need_install ]; then
    echo "One or more packages are not installed, running sudo apt-get install..."
    sudo apt-get install -y $packages
  fi
}

redhat_based_setup() {
  check_dist "$1" "$2"
  packages=$(cat "$root/$3" | tail -n +9 | sed -r -e 's/^\s+//g' -e 's/\\//g' | tr '\n' ' ')
  need_install=""
  for package in $packages; do
    if ! yum list installed | grep "^$package\." > /dev/null; then
      echo "$package is not installed"
      need_install="yes"
    fi
  done
  if [ ! -z $need_install ]; then
    echo "One or more packages are not installed, running su -c \"yum install package0 package1 ...\""
    su -c "yum install -y $packages"
  fi
}

setup_distribution() {
  which lsb_release > /dev/null || \
    { echo "lsb_release was not found => unable to determine your Linux distribution" 1>&2 ; exit 1; }

  dist_id=$(lsb_release -i | cut -d : -f 2 | tr -d "\t")
  case "$dist_id" in
  "Ubuntu"):
      debian_based_setup "Ubuntu" 14 "docs/source/manualrst_veles_user_ubuntu_packages.rst"
      ;;
  "CentOS"):
      redhat_based_setup "CentOS" 6 "docs/source/manualrst_veles_user_centos_packages.rst"
      ;;
  "Fedora"):
      redhat_based_setup "Fedora" 20 "docs/source/manualrst_veles_user_fedora_packages.rst"
      ;;
  *) echo "Did not recognize your distribution \"$dist_id\"" 1>&2
     echo "This is possibly because LSB package is not installed on your system."
     echo "On Debian/Ubuntu: sudo apt-get install lsb-core"
     echo "On CentOS/RHEL/Fedora: su -c yum install redhat-lsb-core"
     exit 1
     ;;
  esac
}

do_post() {
  setup_distribution
  cd $path
  export PYENV_ROOT=$path/pyenv
  . ./init-pyenv
  versions=$(pyenv versions | grep $PYVER || true)
  if [ -z "$versions" ]; then
    pyenv install $PYVER
  fi
  pyenv global $PYVER

  vroot=$path/pyenv/versions/$PYVER
  if [ ! -e $vroot/lib/libsodium.so ]; then
    git clone https://github.com/jedisct1/libsodium
    cd libsodium
    git checkout 1.0.0
    mkdir build
    patch configure.ac < ../libsodium.patch
    ./autogen.sh && cd build
    ../configure --prefix=$vroot --disable-static
    make -j$cpus && make install
    cd ../.. && rm -rf libsodium
  fi

  if [ ! -e $vroot/lib/libpgm.so ]; then
    svn checkout http://openpgm.googlecode.com/svn/trunk/ openpgm
    cd openpgm/openpgm/pgm && mkdir build && mkdir m4
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
  twisted_ver="0.0.0"
  pip3 freeze | grep Twisted > /dev/null && twisted_ver=$(pip3 freeze | grep Twisted | cut -d "=" -f 3)
  if [ "$twisted_ver" \< "14.0.0" ]; then
    pip3 install git+https://github.com/vmarkovtsev/twisted.git
  fi

  # install patched matplotlib v1.4.0
  mpl_ver="0.0.0"
  pip3 freeze | grep matplotlib > /dev/null && mpl_ver=$(pip3 freeze | grep matplotlib | cut -d "=" -f 3)
  if [ "$mpl_ver" \< "1.4.0" ]; then
    if [ ! -e "matplotlib" ]; then
      git clone -b v1.4.1rc1 https://github.com/matplotlib/matplotlib.git
    fi
    pip3 install numpy==1.8.2
    pip3 install -e ./matplotlib
  fi

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
