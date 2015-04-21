#!/usr/bin/env python3
# encoding: utf-8
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Sept 14, 2014

This scripts generates the HTML page with all velescli's command line
arguments, allowing for fast command line composition

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""

import argparse
import gc
from inspect import getargspec
import json
import os
import sys
import threading
import tornado.template as template
import warnings

from veles.config import root
from veles.cmdline import CommandLineBase
from veles.import_file import try_to_import_file, is_module
from veles.unit_registry import UnitRegistry


WEB_FOLDER = root.common.web.root
BLACKLISTED_DIRS = {"external", "znicz/external", "tests", "znicz/tests/unit",
                    "znicz/tests/functional"}

BLACKLISTED_FILES = {"boxer.py", "bbox_detection.py", "generate_frontend.py",
                     "__init__.py"}


def main():
    debug_imports = bool(sys.argv[1]) if len(sys.argv) > 1 else False
    print("Scanning for workflow files...")
    list_workflows = scan_workflows(debug_imports)

    print("Processing arguments...")
    parser = CommandLineBase.init_parser(ignore_conflicts=True)
    arguments = parser._actions
    list_lines = [obj[-1] for obj in sorted(
        [convert_argument(*ia) for ia in enumerate(arguments)])]

    defaults = {}
    html = ''.join(list_lines)
    loader = template.Loader(os.path.join(WEB_FOLDER, "templates"))
    opts, positional_opts, choices, defaults_opt = generate_opts(arguments)

    path_to_out = os.path.join(WEB_FOLDER, "frontend.html")
    print("Writing %s..." % path_to_out)
    sout = loader.load("frontend.html").generate(
        arguments=html, workflows=list_workflows,
        cmdline_states=json.dumps(defaults),
        opts=json.dumps(opts), positional_opts=json.dumps(positional_opts),
        choices=json.dumps(choices), defaults_opt=json.dumps(defaults_opt),
        special_opts=json.dumps(CommandLineBase.SPECIAL_OPTS))
    with open(path_to_out, "wb") as fout:
        fout.write(sout)
    return 0


def scan_workflows(debug_imports):
    workflows = []
    warnings.simplefilter("ignore")
    UnitRegistry.enabled = False
    root_dir = root.common.veles_dir
    for path, _, files in os.walk(root_dir, followlinks=True):
        relpath = os.path.relpath(path, root_dir)
        skip = False
        for bdir in BLACKLISTED_DIRS:
            if relpath.startswith(bdir):
                skip = True
                break
        if skip:
            continue
        for f in set(files) - BLACKLISTED_FILES:
            f_path = os.path.join(path, f)
            modname, ext = os.path.splitext(f)
            if ext != '.py':
                continue
            if debug_imports:
                sys.stdout.write("[%s] importing %s... " % (relpath, modname))
            mod = try_to_import_file(f_path)
            if not is_module(mod):
                if debug_imports:
                    print("SKIP (import)")
                continue
            for func in dir(mod):
                if func == "run" and \
                        getargspec(mod.run).args == ["load", "main"]:
                    wf_path = os.path.relpath(f_path, root_dir)
                    workflows.append(wf_path)
                    if debug_imports:
                        print("OK")
                    break
            else:
                if debug_imports:
                    print("SKIP (run)")
    gc.collect()
    warnings.simplefilter("default")
    print("Found %d workflows:\n" % len(workflows), workflows)
    if len(threading.enumerate()) > 1:
        print("Warning: more than 1 thread is currently running, a join lock "
              "may happen.")
    return workflows


def generate_opts(arguments):
    string_arguments = []
    boolean_arguments = []
    positional_opts = []
    alias = {}
    choices = {}
    defaults = {}
    for arg in arguments:
        if arg.option_strings:
            option_strings = arg.option_strings
            option = str(option_strings[:2][-1])
        else:
            positional_opts.append(arg.dest)
            option = arg.dest
        if arg.choices is not None:
            choice = list(arg.choices)
            choices[option] = choice
        defaults[option] = arg.default
        while option[0] == "-":
            option = option[1:]
        if isinstance(arg, argparse._StoreTrueAction):
            boolean_arguments.append(option)
        elif isinstance(arg, argparse._StoreAction):
            string_arguments.append(option)
        if len(option_strings) > 1:
            for i in range(0, len(option_strings), 2):
                while option_strings[i][0] == "-":
                    option_strings[i] = option_strings[i][1:]
                while option_strings[i + 1][0] == "-":
                    option_strings[i + 1] = option_strings[i + 1][1:]
                alias[option_strings[i]] = option_strings[i + 1]
    opts = {"string": string_arguments,
            "alias": alias,
            "boolean": boolean_arguments,
            "stopEarly": True}
    return opts, positional_opts, choices, defaults


def convert_argument(index, arg):
    index += 1
    choices = arg.choices
    nargs = arg.nargs
    required = arg.required and nargs != '*'
    if len(arg.help) > 0:
        if arg.help[-1] != '.':
            arg.help += '.'
        arg.help = arg.help[0].capitalize() + arg.help[1:]
        # we insert %% instead of % to work around argparse bug
        arg.help = arg.help.replace("%%", "%")
    arg_mode = getattr(arg, "mode", ["standalone", "master", "slave"])
    arg_line = ""
    if arg.option_strings:
        title = ", ".join(arg.option_strings)
        key = arg.option_strings[-1]
        option_strings = str(arg.option_strings[:2][-1])
    else:
        title = getattr(arg, "pretty_name", arg.dest)
        key = title
        option_strings = arg.dest
    if arg.dest == "workflow":
        arg_line = convert_workflow(arg, arg_mode, option_strings)
    else:
        if choices is not None:
            arg_line = convert_choices(arg, arg_mode, option_strings)
        else:
            if isinstance(arg, argparse._StoreTrueAction):
                arg_line = convert_boolean(arg, arg_mode, option_strings)
            if isinstance(arg, argparse._StoreAction):
                arg_line = convert_string(arg, arg_mode, option_strings)
    imp = -float(required) / index
    if hasattr(arg, "important"):
        imp = -float(arg.important) / index
    importance = 'Mandatory' if required else 'Optional'
    importance_class = 'danger' if required else 'default'
    template_line = """
            <div class="panel panel-primary argument %s">
                <div class="panel-heading %s">
                  <span class="label label-%s argtype">%s</span>
                  <h3 class="panel-title">%s</h3>
                </div>
                <div class="panel-body">
                    <div class="pull-right description">
                      <p>%s</p>
                    </div>
                    %s
                </div>
            </div>""" % (" ".join(arg_mode), " ".join(arg_mode),
                         importance_class, importance, title, arg.help,
                         arg_line)
    return (imp, key, template_line)


def convert_workflow(arg, arg_mode, option_strings):
    dest = arg.dest
    default = arg.default
    arg_line = ("""
                    <div class="input-group" id = "scrollable-dropdown-menu">
                     <span class="input-group-addon">%s</span>
                     <input type="text" class="typeahead form-control %s"
                      placeholder="%s" id="%s">
                    </div>""" % (dest, " ".join(arg_mode), default,
                                 option_strings))
    return arg_line


def convert_string(arg, arg_mode, option_strings):
    dest = arg.dest
    default = arg.default
    arg_line = ("""
                    <div class="input-group">
                     <span class="input-group-addon">%s</span>
                     <input type="text" class="form-control %s"
                      placeholder="%s" id="%s">
                    </div>""" % (dest, " ".join(arg_mode), default,
                                 option_strings))
    return arg_line


def convert_boolean(arg, arg_mode, option_strings):
    default = arg.default
    checked = "checked" if default else ""
    arg_line = ("""
                    <div class="bootstrap-switch-container">
                      <input type="checkbox" class="switch %s"
                       data-on-text="Yes"
                       data-off-text="No" data-size="large" %sid="%s"/>
                    </div>""" % (" ".join(arg_mode), checked, option_strings))
    return arg_line


def convert_choices(arg, arg_mode, option_strings):
    choices = arg.choices
    choices_lines = ''
    default = arg.default
    for choice in choices:
        line_ch = ("""
            <li role="presentation"><a role="menuitem"tabindex="-1"
            href="javascript:select('%s', '%s')">%s</a></li>""" % (
            choice, option_strings, choice))
        choices_lines += line_ch
    arg_line = ("""
                <div class="dropdown">
                  <button class="btn btn-default dropdown-toggle %s"
                  type="button" id="dropdown_menu%s" data-toggle="dropdown"
                  >
                    %s
                    <span class="caret"></span>
                  </button>
                  <ul class="dropdown-menu %s" role="menu"
                  aria-labelledby="dropdown_menu" id="%s">
                    %s
                  </ul>
                </div>""" % (" ".join(arg_mode), option_strings, default,
                             " ".join(arg_mode),
                             option_strings, choices_lines))
    return arg_line

if __name__ == "__main__":
    sys.exit(main())
