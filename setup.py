# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
from veles import __root__, __project__, __version__, __license__,\
    __authors__, __contact__


MANIFEST = os.path.join(__root__, "MANIFEST.in")


def write_exclusions(gitattrfile):
    root = os.path.relpath(os.path.dirname(gitattrfile), __root__)
    with open(gitattrfile, "r") as fin:
        lines = fin.readlines()
    exclusions = []
    for item in (line[:-len('export-ignore\n')].strip() for line in lines
                 if line.endswith('export-ignore\n')):
        path = os.path.join(root, item)
        if path.startswith("./"):
            path = path[2:]
        if os.path.isdir(path):
            exclusions.append("recursive-exclude %s *\n" % path)
        elif os.path.exists(path):
            exclusions.append("exclude %s\n" % path)
        else:
            parent = os.path.dirname(path)
            if parent:
                mask = os.path.basename(item)
                exclusions.append("recursive-exclude %s %s\n" % (parent, mask))
            else:
                exclusions.append("exclude %s\n" % path)
    with open(MANIFEST, "a") as fout:
        fout.writelines(exclusions)

if __name__ == "__main__":
    if not os.path.exists(MANIFEST):
        print("processing .gitattributes")
        if os.path.exists(MANIFEST):
            os.remove(MANIFEST)
        for root, dirs, files in os.walk(__root__):
            if ".gitattributes" in files:
                write_exclusions(os.path.join(root, ".gitattributes"))
        with open(MANIFEST, "a") as fout:
            fout.write("recursive-exclude deploy *\n")
            fout.write("exclude ubuntu-apt-get-install-me.txt\n")
            fout.write("exclude MANIFEST.in\n")
            fout.write("recursive-include docs/build/html *\n")

    setup(
        setup_requires=["setuptools_git", ],
        name="Veles",
        description=__project__,
        version=__version__,
        license=__license__,
        author=__authors__,
        author_email=__contact__,
        url="http://confluence.rnd.samsung.ru/display/VEL/Veles",
        download_url="http://alserver.rnd.samsung.ru/gerrit/Veles",
        packages=find_packages(),
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'veles=veles.__main__:__run__',
                'compare_snapshots=veles.scripts.compare_snapshots.main',
                'bboxer=veles.scripts.bboxer.main',
                'generate_frontend_veles=veles.scripts.generate_frontend.main']
        },
        keywords=["Samsung", "Veles", "Znicz"],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: " + __license__,
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: Implementation :: PyPy",
            "Topic :: Scientific/Engineering :: Information Analysis"
        ]
    )