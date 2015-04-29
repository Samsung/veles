# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
import veles
from veles import __root__, __project__, __version__, __license__,\
    __authors__, __contact__, __versioninfo__, __date__, formatdate


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
            exclusions.append("prune %s\n" % path)
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


def make_release(patch_init=True, patch_changelog=True):
    incver = __versioninfo__[:-1] + (__versioninfo__[-1] + 1,)
    strver = [str(v) for v in incver]
    if patch_init:
        with open(veles.__file__, "r+") as init:
            patched = init.read().replace(
                "__versioninfo__ = %s" % ", ".join(map(str, __versioninfo__)),
                "__versioninfo__ = %s" % ", ".join(strver))
            init.seek(0, os.SEEK_SET)
            init.write(patched)
    chlog_head = """veles (%(ver)s-0) trusty utopic; urgency=medium

  * %(ver)s release, see https://github.com/samsung/veles/releases/tag/v%(ver)s

 -- Markovtsev Vadim <v.markovtsev@samsung.com>  %(date)s

""" % {"ver": ".".join(strver), "date": formatdate(__date__, True)}
    if patch_changelog:
        with open(os.path.join(__root__, "debian", "changelog"), "r+") as fch:
            changelog = fch.read()
            changelog = chlog_head + changelog
            fch.seek(0, os.SEEK_SET)
            fch.write(changelog)
    return incver


if __name__ == "__main__":
    if not os.path.exists(MANIFEST):
        print("processing .gitattributes")
        if os.path.exists(MANIFEST):
            os.remove(MANIFEST)
        for root, dirs, files in os.walk(__root__):
            if ".gitattributes" in files:
                write_exclusions(os.path.join(root, ".gitattributes"))
        with open(MANIFEST, "a") as fout:
            fout.write("prune deploy\n")
            fout.write("exclude MANIFEST.in\n")

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
                'compare_snapshots=veles.scripts.compare_snapshots:main',
                'bboxer=veles.scripts.bboxer:main',
                'generate_veles_frontend=veles.scripts.generate_frontend:main',
                'veles_graphics_client=veles.graphics_client:main']
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