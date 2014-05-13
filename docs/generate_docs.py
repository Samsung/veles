#!/usr/bin/python3
# encoding: utf-8
"""
This scripts generates Veles project documentation
To generate the docs, you have to install sphinx3-doc and the extensions:
    sphinxcontrib-napoleon, sphinx.ext.mathjax and sphinx.ext.argparse
"""

import os
import subprocess
import sys

docs_path = os.path.dirname(os.path.realpath(__file__))
docs_source_path = os.path.join(docs_path, "source")
project_path = os.path.realpath(os.path.join(docs_path, ".."))

sys.path.append(docs_path)

subprocess.call(["sphinx-apidoc", "-f", "-o", docs_source_path, project_path])
subprocess.call(["make", "html"])
