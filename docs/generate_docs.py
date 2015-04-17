#!/usr/bin/python3
# encoding: utf-8
"""
This scripts generates Veles project documentation.
To generate the docs, you have to install sphinx3-doc and the extensions:
    sphinxcontrib-napoleon, sphinx.ext.mathjax and sphinx.ext.argparse
"""

import argparse
import os
import subprocess
import sys

try:
    from .generate_units_args import UnitsKeywordArgumentsGenerator, \
        docs_source_dir, result_file_name_base
except SystemError:
    from generate_units_args import UnitsKeywordArgumentsGenerator, \
        docs_source_dir, result_file_name_base


def main():
    parser = argparse.ArgumentParser(description="VELES documentation builder")
    parser.add_argument("--skip-deps", default=False,
                        help="Do not generate the module dependency diagrams.",
                        action='store_true')
    parser.add_argument("--skip-units-kwargs", default=False,
                        help="Do not generate the keyword arguments list for "
                             "all units.",
                        action='store_true')
    parser.add_argument("--skip-build", default=False,
                        help="Do not generate the documentation with Sphinx.",
                        action='store_true')
    args = parser.parse_args()
    docs_path = os.path.dirname(os.path.realpath(__file__))
    docs_source_path = os.path.join(docs_path, "source")
    project_path = os.path.realpath(os.path.join(docs_path, ".."))

    sys.path.append(docs_path)
    os.chdir(docs_path)

    if not args.skip_deps:
        subprocess.call(["./deps.sh"])
    if not args.skip_units_kwargs:
        UnitsKeywordArgumentsGenerator().run(
            os.path.join(docs_source_dir, result_file_name_base + ".rst"))
    if args.skip_build:
        return
    subprocess.call([
        "sphinx-apidoc", "-e", "-f", "-H", "Source Code", "-o",
        docs_source_path, project_path,
        os.path.join(project_path, "docs"),
        os.path.join(project_path, "veles/external"),
        os.path.join(project_path, "veles/znicz/external"),
        os.path.join(project_path, "veles/tests"),
        os.path.join(project_path, "veles/znicz/tests")])
    os.remove("source/modules.rst")
    os.remove("source/setup.rst")
    subprocess.call(["sphinx-build", "-b", "html", "-d",
                     os.path.join(docs_path, "build/doctrees"),
                     docs_source_path, os.path.join(docs_path, "build/html")])

if __name__ == "__main__":
    main()
