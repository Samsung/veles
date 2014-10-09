============
OpenCL setup
============

In order to run Veles in OpenCL mode, there must be an OpenCL driver properly
installed on your system.

Nvidia OpenCL Driver installation
:::::::::::::::::::::::::::::::::

Change to tty1 terminal (CTRL+ALT+F1) and stop X server (this will terminate all your running applications!)::

    sudo service lightdm stop

Copy the installation .run file from /data/veles/packages/binary and execute it
or use the following shell script::

    #!/bin/bash -e
    
    if [[ -z "$1" ]]; then
      echo "You must specify the driver version, e.g., \"331.67\"" 1>&2
      exit 1
    fi
    
    package=NVIDIA-Linux-x86_64-$1.run
    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/$1/$package
    chmod +x $package
    ./$package -a -q -l --no-opengl-files
      
This script requires a single command line argument which sets the driver version.
Refer to `Nvidia's web site <http://nvidia.com>`_ to find out what version is the latest.
Anyway, "-l" argument makes the underlying Nvidia installation script to install
the most recent version available, whereas "-a" accepts the license and "-q"
answers "yes" on all questions. "--no-opengl-files" is strictly necessary
because by default Nvidia's installer writes OpenGL libs which conflict with integrated
video OpenGL libs and Unity fails to start.

Start the X server::

   sudo service lightdm start

As for CUDA installation, it should be straightforward, but pay attention to
**not install driver during cuda installation process**.

Nvidia drivers troubleshooting
::::::::::::::::::::::::::::::

^^^^^^^
nouveau
^^^^^^^

In case of drivers problem check that there is no conflicts with `nouveau` driver::

   sudo modprobe nvidia

If an error appears , e.g. "modprobe: ERROR: could not insert 'nvidia': No such device"
check `nouveau` driver presence::

   lsmod | grep nouveau

If nouveau is found, add it to the black list::

   echo "blacklist nouveau" | sudo tee -a /etc/modprobe.d/blacklist-framebuffer.conf

Then reinstall Nvidia drivers package.

^^^^^^^^^^^^^^^^
driver unloading
^^^^^^^^^^^^^^^^

Looped ``nvidia-smi`` prevents from driver unloading::

   nohup nvidia-smi -l