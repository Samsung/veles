Licenses of packages which Veles depends on
===========================================

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
| tornado               | Apache License, Version 2.0                |
| motor                 | Apache License, Version 2.0                |
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
| lockfile              | MIT                                        |
| python-snappy         | BSD                                        |
| pycrypto              | Public domain                              |
| ecdsa                 | MIT                                        |
| pyzmq                 | LGPL, BSD -> BSD                           |
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
| pyodbc                | MIT                                        |
| progressbar (bundled) | LGPL, BSD -> BSD                           |
| fysom (bundled)       | BSD                                        |
| freetype (bundled)    | BSD                                        |
| daemon (bundled)      | PSF                                        |
| freetype (bundled)    | BSD                                        |
| pydot (bundled)       | MIT                                        |
| manhole (bundled)     | BSD                                        |
| prettytable (bundled) | BSD                                        |
| pytrie (bundled)      | BSD                                        |
| txzmq (rewritten)     | Apache License, Version 2.0 (was GPL)      |
| xmltodict (bundled)   | BSD                                        |
