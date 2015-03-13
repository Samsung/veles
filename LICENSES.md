Licenses of packages Veles depends on
=====================================

The table below was filled using the data obtained via the following script:

```sh
pip install git+https://github.com/vmarkovtsev/yolk.git
for d in $(cat requirements.txt); do yolk -f license,name -M $(echo "$d" | sed 's/[<>=]/ /g' | cut -d ' ' -f 1) && echo ""; done
```

| name                  | license                                    |
|-----------------------|--------------------------------------------|
| matplotlib            | BSD                                        |
| numpy                 | BSD                                        |
| scipy                 | BSD                                        |
| Pillow                | Standard PIL License                       |
| six                   | MIT                                        |
| tornado               | http://www.apache.org/licenses/LICENSE-2.0 |
| motor                 | http://www.apache.org/licenses/LICENSE-2.0 |
| pymongo               | Apache License, Version 2.0                |
| Twisted               | MIT                                        |
| ply                   | BSD                                        |
| paramiko              | LGPL                                       |
| opencl4py             | Simplified BSD                             |
| argcomplete           | Apache Software License                    |
| ipython               | BSD                                        |
| jpeg4py               | Simplified BSD                             |
| cffi                  | MIT                                        |
| Glymur                | MIT                                        |
| lockfile              | None                                       |
| python-snappy         | BSD                                        |
| pycrypto              | Public domain                              |
| ecdsa                 | MIT                                        |
| pyzmq                 | LGPL+BSD                                   |
| wget                  | Public Domain                              |
| service_identity      | MIT                                        |
| pygit2                | GPLv2 with linking exception               |
| pyinotify             | MIT License                                |
| cuda4py               | Simplified BSD                             |
| psutil                | BSD                                        |
| pyxDamerauLevenshtein | BSD 3-Clause License                       |
| h5py                  | BSD                                        |
| Jinja2                | BSD                                        |
| pycparser             | BSD                                        |
