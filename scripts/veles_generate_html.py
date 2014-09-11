#!/usr/bin/python3
# encoding: utf-8
'''
This scripts generates an HTML page with all velescli's command line arguments,
 allowing for fast command line composition

'''


import argparse
import gc
from inspect import getargspec
import json
import os
import sys
from scripts.velescli import Main
import tornado.template as template
from veles.config import root
import warnings

WEB_FOLDER = root.common.web_folder


def main():
    parser = Main.init_parser()
    arguments = parser._actions
    path_to_out = os.path.join(WEB_FOLDER, "frontend.html")
    list_lines = []
    list_workflows = []
    root_path = root.common.veles_dir
    warnings.simplefilter("ignore")
    for path, _, files in os.walk(root_path, followlinks=True):
        if os.path.relpath(path, root_path).startswith('docs'):
            continue
        for f in files:
            f_path = os.path.join(path, f)
            modname, ext = os.path.splitext(f)
            if ext == '.py':
                sys.path.insert(0, path)
                try:
                    mod = __import__(modname)
                    for func in dir(mod):
                        if func == "run":
                            if getargspec(mod.run).args == ["load", "main"]:
                                wf_path = os.path.relpath(f_path, root_path)
                                list_workflows.append(wf_path)
                except:
                    pass
                finally:
                    del sys.path[0]
    gc.collect()
    warnings.simplefilter("default")
    for tuple_obj in sorted([convert_argument(arg) for arg in arguments]):
        list_lines.append(tuple_obj[1])
    defaults = {}
    html = ''.join(list_lines)
    loader = template.Loader(os.path.join(WEB_FOLDER, "templates"))
    sout = loader.load("frontend.html").generate(
        arguments=html, workflows=list_workflows,
        initial_states=json.dumps(defaults))
    with open(path_to_out, "wb") as fout:
        fout.write(sout)


def convert_argument(arg):
    choices = arg.choices
    nargs = arg.nargs
    required = arg.required and nargs != '*'
    dest = arg.dest
    hlp = arg.help
    arg_mode = getattr(arg, "mode", ["standalone", "master", "slave"])
    arg_line = ""
    if arg.option_strings:
        option_strings = str(arg.option_strings[0])
    else:
        option_strings = arg.dest
    if dest == "workflow":
        arg_line = convert_workflow(arg, arg_mode, option_strings)
    else:
        if choices is not None:
            arg_line = convert_choices(arg, arg_mode, option_strings)
        else:
            if isinstance(arg, argparse._StoreTrueAction):
                arg_line = convert_boolean(arg, arg_mode, option_strings)
            if isinstance(arg, argparse._StoreAction):
                arg_line = convert_string(arg, arg_mode, option_strings)
    imp = int(not required)
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
                      <p>%s.</p>
                    </div>
                    %s
                </div>
            </div>""" % (" ".join(arg_mode), " ".join(arg_mode),
                         importance_class, importance, dest, hlp, arg_line)
    return (imp, template_line)


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
                        href="#" onclick="select('%s', '%s')">%s</a></li>""" %
                        (choice, option_strings, choice))
        choices_lines += line_ch
        arg_line = ("""
                    <div class="dropdown">
                      <button class="btn btn-default dropdown-toggle %s"
                      type="button" id="dropdown_menu" data-toggle="dropdown">
                        %s
                        <span class="caret"></span>
                      </button>
                      <ul class="dropdown-menu" role="menu"
                      aria-labelledby="dropdown_menu" id="%s">
                        %s
                      </ul>
                    </div>""" % (" ".join(arg_mode), default,
                                 option_strings, choices_lines))
    return arg_line

if __name__ == "__main__":
    retcode = main()
    sys.exit(retcode)
